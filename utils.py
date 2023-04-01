# Default parameters used in the ViT architecture

DROPOUT_PROB = 0.1  # probability of applying dropout in the MLP module
BATCH_SIZE = 8  # how many images to put in a batch
INPUT_CH = 3  # number of channels of the input images, default RGB 3 color images
INPUT_H = 224  # number of pixels in the height of the images, default 224 ImageNet style
INPUT_W = 224  # number of pixels in the width of the images, default 224 ImageNet style
NUM_CLASSES = 3  # number of output classes

PATCH_SIZE = 16  # height and width of the image patches
EMBEDDING_DIM = 768  # size of linear projections of the image patches
NUM_HEADS = 8  # number of heads to use in the multi-headed attention module
MLP_NEURONS = 2048  # number of neurons to use in the hidden layer of the MLP module
NUM_LAYERS = 6  # number of encoder layers to use in the ViT
NUM_PATCHES = (INPUT_H // PATCH_SIZE) * (INPUT_W // PATCH_SIZE)  # total number of patches we get from one image
