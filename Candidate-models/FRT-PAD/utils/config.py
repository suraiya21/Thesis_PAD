#encoding:utf-8
class DefaultConfig(object):

	root ='./'
	Epoch_num = 15
	batch_size = 16
	eval_epoch = 1
	sample_frame = 1

	opt = 'Adam'
	learning_rate = 1e-4
	momentum = 0.9
	weight_decay = 5e-5

	shuffle_train = True

	dataset = {
	'mobai_train': 'Mobai_Train',
	'mobai_test': 'Mobai_Test'
	}

	face_related_work={
	'FR':'Face_Recognition',
	'FE':'Face_Expression_Recognition',
	'FA':'Face_Attribute_Editing'
	}

	graph={
	'direct': 'Step_by_Step_Graph',
	'dense': 'Dense_Graph'
	}
	savedir = './save_models/'