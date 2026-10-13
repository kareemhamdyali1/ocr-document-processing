# utils/pdf_to_images.py

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_document(path: str) -> list[Image.Image]:
    """
    Load an image or PDF and return a list of RGB PIL images.

    Supported:
        PDF, JPG, JPEG, PNG, WEBP, GIF, BMP, TIF, TIFF

    PDF files require pdf2image and Poppler.
    """
    file_path = Path(path).expanduser()

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Document path is not a file: {file_path}"
        )

    suffix = file_path.suffix.lower()

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path

        except ImportError as exc:
            raise ImportError(
                "PDF support requires pdf2image.\n"
                "Install it with: pip install pdf2image\n"
                "Poppler must also be installed and available."
            ) from exc

        try:
            pages = convert_from_path(
                str(file_path),
                dpi=300,
                fmt="png",
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert PDF to images: {file_path}\n"
                f"Original error: {exc}\n"
                "Check the PDF file and Poppler installation."
            ) from exc

        if not pages:
            raise ValueError(
                f"The PDF contains no readable pages: {file_path}"
            )

        converted_pages = []

        for index, page in enumerate(pages, start=1):
            try:
                rgb_page = ImageOps.exif_transpose(
                    page
                ).convert("RGB")

                converted_pages.append(
                    rgb_page.copy()
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to process PDF page {index}: {exc}"
                ) from exc

            finally:
                try:
                    page.close()
                except Exception:
                    pass

        return converted_pages

    # -----------------------------------------------------
    # Image files
    # -----------------------------------------------------

    if suffix in IMAGE_SUFFIXES:
        try:
            with Image.open(file_path) as image:
                # For animated images, use the first frame.
                try:
                    image.seek(0)
                except EOFError:
                    pass

                image = ImageOps.exif_transpose(
                    image
                ).convert("RGB")

                return [image.copy()]

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Unable to open image: {file_path}\n"
                f"Original error: {exc}"
            ) from exc

    raise ValueError(
        f"Unsupported file extension: {suffix}\n"
        "Supported extensions: PDF, JPG, JPEG, PNG, WEBP, "
        "GIF, BMP, TIF, TIFF."
    )