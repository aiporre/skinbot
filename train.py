
from skinbot.dataset import WoundImages
from skinbot.config import read_config

def main():
    config = read_config()
    root_dir = config["DATASET"]["root"]
    wound_images = WoundImages(root_dir)
    
    for cnt, (img, label) in enumerate(wound_images):
        print("read: ", label, "image path ", wound_images.image_fnames[cnt])

if __name__ == "__main__":
    main()
    

