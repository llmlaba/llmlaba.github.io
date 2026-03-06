---
layout: default
title: "Uploading Custom GGUF Models to Hugging Face"
date: 2026-03-06
categories: [general]
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Uploading Custom GGUF Models to Hugging Face

This guide explains how to correctly upload custom `.gguf` models to a repository on Hugging Face using **Git LFS**.

Large model files (like `.gguf`) cannot be pushed using normal Git because Hugging Face limits regular files to **10 MB**. Instead, they must be stored using **Git LFS (Large File Storage)**.

## 1. Install Git LFS

Install Git LFS if you don't already have it.

- Linux

```bash
sudo apt install git-lfs
```
- Mac

```bash
brew install git-lfs
```
- Windows

Download and install from:
https://git-lfs.github.com/

- Initialize Git LFS:

```bash
git lfs install
```

## Install huggingface-cli

- Mac

```bash
brew install huggingface-cli
```

## 2. Create a New Hugging Face Repository

Create a model repository on Hugging Face.

Example repository name:

```
username/my-gguf-model
```

Clone the repository:

```bash
git clone https://huggingface.co/username/my-gguf-model
cd my-gguf-model
```

- Enable large file support for the repository:
> **Required!** Without this step, Hugging Face will reject files larger than 5 GB.

```bash
hf lfs-enable-largefiles ./
```

---

## 3. Track GGUF Files with Git LFS

- Tell Git to store `.gguf` files using Git LFS:
> This creates a `.gitattributes` file.

```bash
git lfs track "*.gguf"
```
- Commit it:

```bash
git add .gitattributes
git commit -m "Enable Git LFS for GGUF models"
```

## 4. Add Model Files

- Copy your `.gguf` model files into the repository folder.
> Example:

```
my-gguf-model/
│
├── README.md
├── model-Q4_K.gguf
├── model-Q8_0.gguf
└── model-bf16.gguf
```
- Add files:

```bash
git add *.gguf
git commit -m "Add GGUF model files"
```

## 5. Push to Hugging Face

- Push the repository:

```bash
git push
```
- You should see something like:

```text
Uploading LFS objects: 100% (3/3), 12 GB
```

## 6. Verify Files

- Check that files are tracked by Git LFS:

```bash
git lfs ls-files
```
- Example output:

```
model-Q4_K.gguf
model-Q8_0.gguf
model-bf16.gguf
```

## 7. Updating Models Later

- To upload new versions or additional quantizations:
> Git LFS will automatically upload the large files.

```bash
git add new-model.gguf
git commit -m "Add new quantization"
git push
```

## Notes

- Always track `.gguf` files with Git LFS before committing.
- Do not push `.gguf` files using regular Git.
- Hugging Face automatically handles large model storage via Git LFS.

Hugging Face documentation:
https://huggingface.co/docs/hub/repositories-getting-started
