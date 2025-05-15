import os
import gzip
import shutil
import time
import json
import h5py
import docker
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mri_reconstruction import process_hdf5_kspace  # Assuming this function is in reconstruction.py

# Constants
KOMA_SERVICE = 'http://127.0.0.1:6000/koma'


def upload_file(address, filetype, filepath):
    """Upload a file to a given address with a specified content type."""
    print(f'Uploading {filepath} to {address}')
    try:
        with open(filepath, 'rb') as file:
            response = requests.post(address, headers={'Content-Type': filetype}, data=file.read())
        print(f'Response: {response.status_code}')
        return response.status_code == 200
    except Exception as e:
        print(f'Upload failed: {e}')
        return False


def download_file(address, destination):
    """Download a file from a given address and save to a destination."""
    print(f'Downloading from {address} to {destination}')
    try:
        response = requests.get(address)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                file.write(response.content)
            return True
        else:
            print(f'Failed to download: Status {response.status_code}')
    except Exception as e:
        print(f'Download failed: {e}')
    return False


def scan(phantom_path, sequence_path, output_rawdata_path):
    """Upload phantom and sequence, trigger scan, and download raw k-space."""
    if not upload_file(f'{KOMA_SERVICE}/phantom', 'application/octet-stream', phantom_path):
        print('Error: Failed to upload phantom.')
        return False
    if not upload_file(f'{KOMA_SERVICE}/sequence', 'text/plain', sequence_path):
        print('Error: Failed to upload sequence.')
        return False
    if not download_file(f'{KOMA_SERVICE}/scan', output_rawdata_path):
        print('Error: Failed to download raw k-space.')
        return False
    print('Scan completed successfully.')
    return True


def create_ks_weighted_images_optimized(phantoms_dir, seq_dir, kso_dir, rawdata_dir, sorted_ks_dir, recon_dir, image_dir):
    """Pipeline to create k-space and weighted images."""
    # Convert paths to Path objects
    phantoms_dir = Path(phantoms_dir)
    seq_dir = Path(seq_dir)
    kso_dir = Path(kso_dir)
    rawdata_dir = Path(rawdata_dir)
    sorted_ks_dir = Path(sorted_ks_dir)
    recon_dir = Path(recon_dir)
    image_dir = Path(image_dir)

    # Get file lists
    phantom_files = sorted(phantoms_dir.glob('*.h5'))
    seq_files = sorted(seq_dir.glob('*.seq'))
    kso_files = sorted(kso_dir.glob('*.json'))

    assert len(phantom_files) == len(seq_files) == len(kso_files), "Input file counts do not match."

    for i, (phantom_path, seq_path, kso_path) in enumerate(zip(phantom_files, seq_files, kso_files)):
        base_name = phantom_path.stem
        output_npy_path = recon_dir / f'w_{base_name}.npy'

        if output_npy_path.exists():
            print(f'[{i+1}/{len(phantom_files)}] Skipping {phantom_path.name} (already reconstructed).')
            continue

        print(f'[{i+1}/{len(phantom_files)}] Processing: {phantom_path.name}')
        assert base_name == seq_path.stem, f"Mismatch: {phantom_path.name} and {seq_path.name}"

        # Step 1: Scan to produce raw k-space
        raw_ks_path = rawdata_dir / phantom_path.name
        if not scan(phantom_path, seq_path, raw_ks_path):
            continue  # Skip on scan failure

        # Step 2: Verify k-space order ID matches
        phantom_id = raw_ks_path.stem.split('subject')[1]
        kso_id = kso_path.stem.split('subject')[1][:-5]  # Remove tail like _info.json
        assert phantom_id == kso_id, f'ID Mismatch: {phantom_id} vs {kso_id}'

        # Step 3: Sort and reconstruct
        sorted_ks, rec_img = process_hdf5_kspace(raw_ks_path, kso_path)

        # Step 4: Save sorted k-space and reconstruction
        np.save(sorted_ks_dir / f'{base_name}.npy', sorted_ks)
        np.save(output_npy_path, rec_img)

        # Step 5: Save PNG image for visualization
        plt.imshow(rec_img, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(image_dir / f'w_{base_name}.png', bbox_inches='tight', pad_inches=0)
        plt.close()



def running_container(compressed_tar_path, decompressed_tar_path):
    client = docker.from_env()

    with gzip.open(compressed_tar_path, 'rb') as f_in:
        with open(decompressed_tar_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with open(decompressed_tar_path, 'rb') as image_tar:
        image = client.images.load(image_tar.read())[0]

    container = client.containers.run(image.id, detach=True, ports={'6000/tcp': 6000})

    container = client.containers.get(container.id)
    print(container.status)

    return container


if __name__ == '__main__':

    container = running_container('koma_service_v110.tar.gz', 'koma_service_v110.tar')
    time.sleep(5)

    # input
   
    base_folder = r'E:\Dataset'

    contrast = 'PD'
    split_type = 'train'
    phantoms_pth = fr'{base_folder}\{split_type}\{contrast}\phantoms'
    seq_dir = fr'{base_folder}\{split_type}\{contrast}\PS\seq'
    kso_pth = fr'{base_folder}\{split_type}\{contrast}\ps\ks_order'

    #output
    output_ks_raw = fr'{base_folder}\{split_type}\{contrast}\kspace\raw'
    output_ks_sorted = fr'{base_folder}\{split_type}\{contrast}\kspace\sorted'

    output_im = fr'{base_folder}\{split_type}\{contrast}\recon\imges'
    output_rec_npy = fr'{base_folder}\{split_type}\{contrast}\recon\npy'



    create_ks_weighted_images_optimized(phantoms_pth, seq_dir, kso_pth,  output_ks_raw,output_ks_sorted, output_rec_npy, output_im)

    container.stop()