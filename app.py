import streamlit as st
from PIL import Image

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
  
	st.title("Pose Estimation App for Images")
	st.text("Built with gluoncv and Streamlit")
	st.markdown("### [Pose Estimation](https://towardsdatascience.com/human-pose-estimation-simplified-6cfd88542ab3)\
     `            `[Alpha Pose](https://medium.com/beyondminds/an-overview-of-human-pose-estimation-with-deep-learning-d49eb656739b) \
		 [[Paper]](https://arxiv.org/abs/1612.00137)\
     `            `[[View Source]](https://github.com/Hardly-Human/Instance-Segmentation-of-Images)")

	image_file = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])

	if image_file is None:
		st.warning("Upload Image and Run Model")

	if image_file is not None:
		image1 = Image.open(image_file)
		rgb_im = image1.convert('RGB') 
		image = rgb_im.save("saved_image.jpg")
		image_path = "saved_image.jpg"
		st.image(image1)

if __name__== "__main__":
	main()