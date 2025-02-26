# Cleave

Inspired by the work of Liu et.al., 2024 (Kangaroo), we propose a way to distill and evalulate smaller versions of a larger language models. For instance a 20 Layer Qwen2.5 Model distilled from the 36 Layer full size version of the same.

#### TODO List
- [ ] add support for sliding window attention

## Getting Started

This repository provides a template for LLM-based projects with:
- **Docker-based development**
- **Jekyll-based documentation**
- **Hugging Face API integrations**
- **Submodules for external repositories (e.g., Meta's Coconut, LLM2Vec)**
- **Poetry-based dependency management**

## Setup Instructions

1. Clone the repository:
   ```sh
   git clone --recurse-submodules https://github.com/cattomantis/hidden.git
   ```

2. Build Docker Container:
   ```sh
   docker compose up experiments -d
   ```
