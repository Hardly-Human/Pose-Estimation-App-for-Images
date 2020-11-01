import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

@st.cache(allow_output_mutation=True)
def load_model(model_name):
	model = model_zoo.get_model(model_name, pretrained = True)
	return model

def plot_image(detector,pose_net, x, img):
	st.warning("Inferencing from Model..")
	class_IDs, scores, bounding_boxs = detector(x)
	pose_input, upscale_bbox = detector_to_alpha_pose(img,class_IDs,scores,bounding_boxs)
	predicted_heatmap = pose_net(pose_input)
	pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
    
	fig = plt.figure(figsize=(10, 10))
	ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.success("Pose Estimation Successful!! Plotting Image..")
	st.pyplot(plt.show())


def main():
  
	st.title("Pose Estimation App for Images")
	st.text("Built with gluoncv and Streamlit")
	st.markdown("### [Pose Estimation](https://towardsdatascience.com/human-pose-estimation-simplified-6cfd88542ab3)\
     `            `[Alpha Pose](https://medium.com/beyondminds/an-overview-of-human-pose-estimation-with-deep-learning-d49eb656739b) \
	 `			  `[[Paper]](https://arxiv.org/abs/1612.00137)\
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

	if st.button("Run Model"):
		st.warning("Loading Model..🤞")
		detector = load_model('yolo3_mobilenet1.0_coco')
		pose_net = load_model('alpha_pose_resnet101_v1b_coco')
		detector.reset_class(["person"], reuse_weights=['person'])
		st.success("Loaded Model Succesfully!!🤩👍")

		x, img = data.transforms.presets.yolo.load_test(image_path, short=512)
		plot_image(detector,pose_net,x,img)

if __name__== "__main__":
	main()