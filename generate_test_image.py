from PIL import Image, ImageDraw


def create_test_image():
    # Create a 512x512 transparent image
    img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a red circle in the center
    draw.ellipse((156, 156, 356, 356), fill=(255, 0, 0, 255))

    # Save
    img.save("valid_test.png")
    print("Created valid_test.png")


if __name__ == "__main__":
    create_test_image()
