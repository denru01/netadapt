# Define constants.
STRING_SEPARATOR = '.'

# Define layer types.
CONV_LAYER_TYPES = ['Conv2d', 'ConvTranspose2d']
FC_LAYER_TYPES = ['Linear']
BNORM_LAYER_TYPES = ['BatchNorm2d']

# Define data types.
WEIGHTSTRING = 'weight'
BIASSTRING = 'bias'
RUNNING_MEANSTRING = 'running_mean'
RUNNING_VARSTRING = 'running_var'
NUM_BATCHES_TRACKED = 'num_batches_tracked'

# Define keys.
KEY_LAYER_TYPE_STR = 'layer_type_str' #(e.g. Linear, Conv2d)
KEY_IS_DEPTHWISE = 'is_depthwise'
KEY_NUM_IN_CHANNELS = 'num_in_channels'
KEY_NUM_OUT_CHANNELS = 'num_out_channels'
KEY_KERNEL_SIZE = 'kernel_size'
KEY_STRIDE = 'stride'
KEY_PADDING = 'padding'
KEY_GROUPS = 'groups'
KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR = 'before_squared_pixel_shuffle_factor'
KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR = 'after_squared_pixel_shuffle_factor'
KEY_INPUT_FEATURE_MAP_SIZE = 'input_feature_map_size'
KEY_OUTPUT_FEATURE_MAP_SIZE = 'output_feature_map_size'
KEY_MODEL = 'model'
KEY_LATENCY = 'latency'