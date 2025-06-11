# Red Hat Distribution Build Instructions

This directory contains the necessary files to build a Red Hat compatible container image for the llama-stack.

## Prerequisites

- Python >=3.10
- `llama` CLI tool installed: `pip install llama-stack`
- Podman or Docker installed
- Access to the quay.io/aipcc/base-images/cpu:2.0-1749153990 image, access to the quay org is
  required, can be requested with this [how-to](https://gitlab.com/redhat/rhel-ai/rhaiis/containers#quay-and-konflux-access ).

## Generating the Containerfile

The Containerfile is auto-generated from a template. To generate it:

1. Make sure you have the `llama` CLI tool installed
2. Run the build script:
   ```bash
   ./redhat-distribution/build.py
   ```

This will:
- Check for the llama CLI installation
- Generate dependencies using `llama stack build`
- Create a new `Containerfile` with the required dependencies

## Editing the Containerfile

The Containerfile is auto-generated from a template. To edit it, you can modify the template in `redhat-distribution/Containerfile.in` and run the build script again.
NEVER edit the generated `Containerfile` manually.

## Building the Container Image

Once the Containerfile is generated, you can build the image using either Podman or Docker:

### Using Podman
```bash
podman build --platform linux/amd64 -f redhat-distribution/Containerfile -t rh .
```

## Notes

- The generated Containerfile should not be modified manually as it will be overwritten the next time you run the build script
- The image is built for the linux/amd64 platform
- The base image used is `quay.io/aipcc/base-images/cpu:2.0-1749153990`


## Push the image to a registry

```bash
podman push <build-ID> quay.io/opendatahub/llama-stack:rh-distribution
