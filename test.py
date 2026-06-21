import argparse
import numpy as np
import os
import torch
import log
import time
import SimpleITK as sitk

from PIL import Image
from models.dualSwinAE import DualSwinAE
from models.FusionNet import FusionNet
from monai.networks.nets import SwinUNETR

from utils import mkdir, preprocess, label_process
from utils import read_img, postprocess, mask2one_hot, one_hot2mask
from utils import get_image_paths, calculate_accuracy, slice_visualization


index = [i for i in range(23, 32)]


def _tensor(x):
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)
    return torch.FloatTensor(x.copy()).permute(3, 0, 1, 2)


def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k.replace('module.', '')] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_checkpoint_state(model_path, device):
    state = torch.load(model_path, map_location=device)
    if 'model' in state:
        state = state['model']
    state = clean_state_dict(state)
    return state


def load_stage3_models(args, io):
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")

    if args.cuda:
        io.cprint(
            'Using GPUs ' + str(args.gpus) + ',' + ' from ' +
            str(torch.cuda.device_count()) + ' devices available'
        )
        torch.cuda.manual_seed_all(args.seed)
    else:
        io.cprint('Using CPU')

    # =========================
    # model paths
    # =========================
    TransNet_model_path = os.path.join(args.save_path, 'SwinAE/model_best.pth')

    Unified_Fusion_model_path = os.path.join(
        args.save_path, 'Stage3/unified_fusion_model.pth'
    )

    Fusion_Head_model_path = os.path.join(
        args.save_path, 'Stage3/fusion_head.pth'
    )

    Segmentation_model_path = os.path.join(
        args.save_path, 'Stage3/seg_model.pth'
    )

    # =========================
    # load encoder-decoder
    # =========================
    feature_net = DualSwinAE().to(device)

    state_dict = torch.load(TransNet_model_path, map_location=device)
    feature_net.load_state_dict(clean_state_dict(state_dict['model']))

    fusion_head_state = load_checkpoint_state(Fusion_Head_model_path, device)
    feature_net.decoder.load_state_dict(fusion_head_state)

    # =========================
    # load unified fusion module
    # =========================
    model = FusionNet().to(device)

    fusion_state = load_checkpoint_state(Unified_Fusion_model_path, device)
    model.load_state_dict(fusion_state)

    # =========================
    # load segmentation head
    # =========================
    seg_model = SwinUNETR(
        # img_size=(32, 128, 128),
        in_channels=16,
        out_channels=5,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0
    ).to(device)

    seg_state = load_checkpoint_state(Segmentation_model_path, device)
    seg_model.load_state_dict(seg_state)

    feature_net.eval()
    model.eval()
    seg_model.eval()

    return feature_net, model, seg_model, device


def patches2image(seg, patch_size=128):
    image = torch.zeros((5, 48, 240, 240))
    cont = 0
    slice_list = [0, 12]

    for d in range(len(slice_list)):
        i = slice_list[d]
        for h in range(0, 240 - patch_size, 240 - patch_size - 1):
            for w in range(0, 240 - patch_size, 240 - patch_size - 1):
                image[:, i:i + 32, h:h + patch_size, w:w + patch_size] = seg[cont]
                cont += 1

    return image


def get_case_name(path):
    name = os.path.basename(path)

    if name.endswith('.nii.gz'):
        name = name.replace('.nii.gz', '')
    else:
        name = os.path.splitext(name)[0]

    return name


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)

    return np.uint8(img)


def save_source_and_fused_slices(img1, img2, fused, save_dir, prefix):

    mkdir(save_dir)

    for k, slice_idx in enumerate(index):
        if slice_idx >= img1.shape[0] or slice_idx >= img2.shape[0] or slice_idx >= fused.shape[0]:
            print(f"[Warning] skip slice {slice_idx}, out of range.")
            continue

        t1_slice = normalize_to_uint8(img1[slice_idx])
        t2_slice = normalize_to_uint8(img2[slice_idx])
        fused_slice = normalize_to_uint8(fused[slice_idx])

        Image.fromarray(t1_slice).save(
            os.path.join(save_dir, f'{prefix}_slice_{slice_idx}_t1.png')
        )

        Image.fromarray(t2_slice).save(
            os.path.join(save_dir, f'{prefix}_slice_{slice_idx}_t2flair.png')
        )

        Image.fromarray(fused_slice).save(
            os.path.join(save_dir, f'{prefix}_slice_{slice_idx}_fused.png')
        )

        concat = np.concatenate([t1_slice, t2_slice, fused_slice], axis=1)

        Image.fromarray(concat).save(
            os.path.join(save_dir, f'{prefix}_slice_{slice_idx}_concat.png')
        )
def save_source_slices_from_whole(args, case_idx, case_name, save_dir):

    whole_t1_dir = os.path.join(args.root, 'whole', 't1')
    whole_t2_dir = os.path.join(args.root, 'whole', 't2-flair')

    if not os.path.exists(whole_t1_dir) or not os.path.exists(whole_t2_dir):
        return

    whole_t1_paths = get_image_paths(whole_t1_dir)
    whole_t2_paths = get_image_paths(whole_t2_dir)

    if case_idx >= len(whole_t1_paths) or case_idx >= len(whole_t2_paths):
        return

    img1 = read_img(whole_t1_paths[case_idx])
    img2 = read_img(whole_t2_paths[case_idx])

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    for k, slice_idx in enumerate(index):
        if slice_idx >= img1.shape[0] or slice_idx >= img2.shape[0]:
            continue

        t1_slice = img1[slice_idx].astype(np.float32)
        t2_slice = img2[slice_idx].astype(np.float32)

        t1_slice = normalize_to_uint8(t1_slice)
        t2_slice = normalize_to_uint8(t2_slice)

        Image.fromarray(t1_slice).save(
            os.path.join(save_dir, f'{case_name}_slice_{slice_idx}_t1.png')
        )

        Image.fromarray(t2_slice).save(
            os.path.join(save_dir, f'{case_name}_slice_{slice_idx}_t2flair.png')
        )
def run_stage3_seg(io, args):
    # =========================
    # image paths
    # =========================
    patch_t1_dir = os.path.join(args.root, 'patches', 't1')
    patch_t2_dir = os.path.join(args.root, 'patches', 't2-flair')
    label_dir = os.path.join(args.root, 'whole', 'label')

    img_path1 = get_image_paths(patch_t1_dir)
    img_path2 = get_image_paths(patch_t2_dir)
    seg_path = get_image_paths(label_dir) if os.path.exists(label_dir) else []

    img_num = min(len(img_path1), len(img_path2))

    patch_size = 128
    slice_list = [0, 12]
    h_positions = list(range(0, 240 - patch_size, 240 - patch_size - 1))
    w_positions = list(range(0, 240 - patch_size, 240 - patch_size - 1))
    patches_per_case = len(slice_list) * len(h_positions) * len(w_positions)

    case_num = img_num // patches_per_case

    io.cprint(f"Found patch T1 images: {len(img_path1)}")
    io.cprint(f"Found patch T2-FLAIR images: {len(img_path2)}")
    io.cprint(f"Patches per case: {patches_per_case}")
    io.cprint(f"Detected case number: {case_num}")

    if img_num % patches_per_case != 0:
        io.cprint(
            f"[Warning] patch number {img_num} cannot be evenly divided by {patches_per_case}. "
            f"Only first {case_num * patches_per_case} patches will be used."
        )

    seg_result_path = os.path.join(args.result_path, 'Segmentation')
    mkdir(seg_result_path)

    # =========================
    # load models
    # =========================
    feature_net, model, seg_model, device = load_stage3_models(args, io)

    feature_net.eval()
    model.eval()
    seg_model.eval()

    with torch.no_grad():
        total_start = time.time()

        for case_idx in range(case_num):
            start = time.time()

            case_name = f'case_{case_idx + 1}'
            case_save_dir = os.path.join(seg_result_path, case_name)
            mkdir(case_save_dir)

            final_out = []

            start_patch = case_idx * patches_per_case
            end_patch = start_patch + patches_per_case

            io.cprint(
                f"Processing {case_name}: patches [{start_patch}, {end_patch})"
            )

            for patch_idx in range(start_patch, end_patch):
                img1 = read_img(img_path1[patch_idx])
                img2 = read_img(img_path2[patch_idx])
                img1 = preprocess(img1)
                img2 = preprocess(img2)
                img1 = _tensor(img1).unsqueeze(0).to(device)
                img2 = _tensor(img2).unsqueeze(0).to(device)

                # feature extract
                feat1 = feature_net.encoder1(img1)
                feat2 = feature_net.encoder2(img2)

                # unified fusion module
                out_feat = model(feat1, feat2)

                # learned segmentation head
                seg_out = seg_model(out_feat)
                seg_out = torch.softmax(seg_out, dim=1)

                seg_out = seg_out.squeeze(0).detach().cpu()
                final_out.append(seg_out)


            final_out = patches2image(final_out)


            # =========================
            # save segmentation result
            # =========================
            final_mask = final_out.permute(1, 2, 3, 0).numpy()
            final_mask = one_hot2mask(final_mask, palette=[0, 1, 2, 3, 4])

            for k, slice_idx in enumerate(index):
                if slice_idx >= final_mask.shape[0]:
                    continue

                seg_RGB = slice_visualization(final_mask, slice_idx)
                seg_RGB = Image.fromarray(seg_RGB, mode='RGB')
                seg_RGB.save(
                    os.path.join(case_save_dir, f'{case_name}_slice_{slice_idx}_seg_RGB.png')
                )

            final_mask_sitk = sitk.GetImageFromArray(final_mask, isVector=False)
            sitk.WriteImage(
                final_mask_sitk,
                os.path.join(case_save_dir, f'{case_name}_seg_results.nii')
            )
            save_source_slices_from_whole(
                args=args,
                case_idx=case_idx,
                case_name=case_name,
                save_dir=case_save_dir
            )

            used_time = time.time() - start
            io.cprint(f'{case_name} segmentation spend {used_time:.4f} seconds')

        total_time = time.time() - total_start
        io.cprint(f'All segmentation cases spend {total_time:.4f} seconds')



def run_stage3_fusion(io, args):
    # =========================
    # image paths
    # =========================

    whole_t1_dir = os.path.join(args.root, 'whole', 't1')
    whole_t2_dir = os.path.join(args.root, 'whole', 't2-flair')

    direct_t1_dir = os.path.join(args.root, 't1')
    direct_t2_dir = os.path.join(args.root, 't2-flair')

    if os.path.exists(whole_t1_dir) and os.path.exists(whole_t2_dir):
        img_path1 = get_image_paths(whole_t1_dir)
        img_path2 = get_image_paths(whole_t2_dir)
    else:
        img_path1 = get_image_paths(direct_t1_dir)
        img_path2 = get_image_paths(direct_t2_dir)

    img_num = min(len(img_path1), len(img_path2))

    io.cprint(f"Found T1 images: {len(img_path1)}")
    io.cprint(f"Found T2-FLAIR images: {len(img_path2)}")
    io.cprint(f"Use image pairs: {img_num}")

    for p in img_path1:
        io.cprint(f"T1: {p}")
    for p in img_path2:
        io.cprint(f"T2-FLAIR: {p}")

    fusion_result_path = os.path.join(args.result_path, 'Fusion')
    mkdir(fusion_result_path)

    # =========================
    # load models
    # =========================
    feature_net, model, seg_model, device = load_stage3_models(args, io)

    with torch.no_grad():
        for i in range(img_num):
            start = time.time()

            case_name = get_case_name(img_path1[i])
            case_save_dir = os.path.join(fusion_result_path, f'case_{i + 1}_{case_name}')
            mkdir(case_save_dir)

            img1_np = read_img(img_path1[i])
            img2_np = read_img(img_path2[i])

            img1_np = preprocess(img1_np)
            img2_np = preprocess(img2_np)

            img1 = _tensor(img1_np).unsqueeze(0).to(device)
            img2 = _tensor(img2_np).unsqueeze(0).to(device)

            # feature extract
            feat1 = feature_net.encoder1(img1)
            feat2 = feature_net.encoder2(img2)

            # unified fusion module
            out_feat = model(feat1, feat2)

            # learned fusion head
            out = feature_net.decoder(out_feat)

            out = out.squeeze().detach().cpu().numpy()
            out = postprocess(out)

            save_source_and_fused_slices(
                img1=img1_np,
                img2=img2_np,
                fused=out,
                save_dir=case_save_dir,
                prefix=f'case_{i + 1}'
            )

            used_time = time.time() - start
            io.cprint(f'Fusion image {i + 1}/{img_num} spend {used_time:.4f} seconds')

if __name__ == '__main__':

    class Args:
        pass

    args = Args()

    # =========================
    # path settings
    # =========================
    args.exp_name = 'Stage3_Test'
    args.out_path = r'experiments/Fusion_experiments'

    args.root = r'test_img'

    args.save_path = r'train_result'

    args.result_path = r'Res'

    # =========================
    # test mode
    # =========================
    # fusion: fusion test only
    # seg:    segmentation test only
    # both:   fusion and segmentation test
    args.mode = 'both'

    # =========================
    # basic settings
    # =========================
    args.workers = 0
    args.seed = 1
    args.gpus = [0]
    args.batch_size = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    io = log.IOStream(args)
    io.cprint(str(args.__dict__))

    mkdir(args.result_path)

    if args.mode in ['fusion', 'both']:
        run_stage3_fusion(io, args)

    if args.mode in ['seg', 'both']:
        run_stage3_seg(io, args)

    io.cprint("Stage III test finished!")