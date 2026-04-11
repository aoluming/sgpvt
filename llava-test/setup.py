from setuptools import setup, find_packages

setup(
    name="llava",
    version="1.0.0",
    description="LLaVA with extended forgetting / perturbation training (see README.md)",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "deepspeed",
        "accelerate",
        "peft",
        "bitsandbytes",
        "pillow",
        "numpy",
        "tqdm",
    ],
    python_requires=">=3.8",
) 