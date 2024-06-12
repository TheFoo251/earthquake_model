from PIL import Image

IMAGE_PATH = "TEST.png"
MASK_PATH = "TEST_MASK.png"
with Image.open(IMAGE_PATH) as im:
    width, height = im.size
    region = im.crop((0, 0, 32, 32))
    region = region.resize((1024, 1024))
    region.show()
