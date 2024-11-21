import modal

image = modal.Image.debian_slim().pip_install("torch", "transformers", "scikit-learn", "datasets")
local_mount = modal.Mount.from_local_dir(".", remote_path="/root/project")
app = modal.App(image=image, mounts=[local_mount])