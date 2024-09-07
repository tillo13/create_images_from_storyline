# create_storyline_images

## Description

`create_storyline_images` is a Python application designed to generate storyline images using advanced models such as Stable Diffusion, IP-Adapter, and others. The application enriches and enhances generated images through various processes, including facial detection, enhancement, and prompt-based customizations.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Primary Script](#primary-script)
  - [Image Enhancement](#image-enhancement)
  - [Face ID Utilities](#face-id-utilities)
  - [Gather Python Files](#gather-python-files)
- [Configuration](#configuration)
- [Licenses](#licenses)

## Project Structure
- **Number of Python files:** 4
- **Number of directories:** 18

### Directory Structure:
```plaintext
.
├── __pycache__
├── comparisons
├── enhanced_images
├── generated_images
└── incoming_images
    └── ai_generated_users
```

### List of Python file paths:
```plaintext
.\create_from_ollama_storyline.py
.\enhance_image_via_import.py
.\faceid_utils.py
.\gather_pythons.py
```

### Detailed Python Files:

#### .\create_from_ollama_storyline.py:
This is the primary script which catalyzes the whole storytelling image generation process. It processes the storyline from a JSON file, integrates it with user-generated content, and applies the Stable Diffusion and IP-Adapter models for detailed representation.

#### .\enhance_image_via_import.py:
Contains utilities for enhancing images, including denoising, sharpening, and enhancing color and contrast. Uses MTCNN for face detection and dlib for facial landmarks.

#### .\faceid_utils.py:
Provides functionalities for handling face ID processes, including downloading necessary model files and extracting face embeddings.

#### .\gather_pythons.py:
Script to gather all Python files within the specified directories and subdirectories, excluding some predefined directories. Generates a summary file listing the Python files and directory structure.

## Installation

To install the dependencies necessary for running this application, use the following commands:
```bash
pip install -r requirements.txt
```

Ensure you have [Hugging Face transformers](https://huggingface.co/transformers/), [Pillow](https://python-pillow.org/), [OpenCV](https://opencv.org/), and other necessary libraries outlined in the respective Python files.

## Usage

### Primary Script

Run `create_from_ollama_storyline.py` to execute the complete image generation and enhancement workflow:

```bash
python create_from_ollama_storyline.py
```

### Image Enhancement

To enhance images using the `enhance_image_via_import.py` script:
```bash
python enhance_image_via_import.py
```
This will apply various image enhancements like denoising and sharpening.

### Face ID Utilities

`faceid_utils.py` provides helper functions to handle face-related processes. The main functions include downloading necessary files and extracting face embeddings.

### Gather Python Files

To gather and summarize Python files:
```bash
python gather_pythons.py
```
This script will generate a file summarizing the Python files and directory structure.

## Configuration

You can configure various settings in `create_from_ollama_storyline.py`:
- **INCOMING_IMAGES_PATH, AI_GENERATED_USERS_PATH, GENERATED_IMAGES_PATH, ENHANCED_IMAGES_PATH, STORYLINES_PATH**: Set paths for respective directories.
- **NUMBER_OF_LOOPS**: Determines the number of iterations for generating images.
- **SEED, CFG_SCALE, NUMBER_OF_STEPS**: Parameters for controlling model generation settings.
- **TOP_MODELS**: List of top models for selection during the generation process.

## Licenses

This project makes use of several open-source software libraries; please review their respective licenses.

By following this documentation, you should be able to effectively understand, install, configure, and use the `create_storyline_images` application for generating and enhancing storyline images.