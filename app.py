import streamlit as st
from PIL import Image

def main():
  
	st.title("Pose Estimation App for Images")
	st.text("Built with gluoncv and Streamlit")
	st.markdown("### [Pose Estimation](https://towardsdatascience.com/human-pose-estimation-simplified-6cfd88542ab3)\
     `            `[Alpha Pose](https://medium.com/beyondminds/an-overview-of-human-pose-estimation-with-deep-learning-d49eb656739b) \
		 [[Paper]](https://arxiv.org/abs/1612.00137)\
     `            `[[View Source]](https://github.com/Hardly-Human/Instance-Segmentation-of-Images)")


if __name__== "__main__":
	main()