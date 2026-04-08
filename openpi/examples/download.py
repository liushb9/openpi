from openpi.shared import download

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
# print(checkpoint_dir)
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
print(checkpoint_dir)