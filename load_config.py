import json, os, codecs


''' logger '''
import codecs
import sys, torch, os

from datetime import datetime
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

import data_utils as dutils
from logger import logger, LogToFile
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a folder if it does not exist, and return the folder.
def initialise_folder(folder, folder_name):		
	if not os.path.exists(folder):
		os.makedirs(folder)
		logger.info("Created new %s directory %s." % (folder_name, folder))
	return folder

# Check whether a filename exists. Throw an error if it does not. Return the filename if it does.
def check_filename_exists(filename):
	if not os.path.isfile(filename):
		logger.error("%s does not exist." % filename)
		raise IOError()
	return filename

# Dump the config object to a file.

def options_to_text(options):
	return "[" + ", ".join(["%s = %s" % (option, value) for option, value in options.items()]) + "]"


# A config object designed to simplify the process of referring to filenames/folders across multiple locations.
class Config():
	def __init__(self, cf):

		self.MODEL = cf['model']
		self.DATASET  = cf['dataset']

		self.MAX_EPOCHS = cf['max_epochs']
		self.LEARNING_RATE = cf['learning_rate']
		self.BATCH_SIZE = cf['batch_size']
		self.MAX_SENT_LEN = cf['max_sent_len']
		self.STOP_CONDITION = cf['stop_condition'] # Stop after this many epochs with no f1 improvement
		self.MAX_SENTS = {"train": cf['max_train_sents'], "test": cf['max_test_sents'], "dev": cf['max_dev_sents']}

		self.EMBEDDING_MODEL = cf['embedding_model']

		self.TASK = cf['task'] # mention_level or end_to_end
		if self.TASK not in ['end_to_end', 'mention_level']:
			logger.error("Task must be either end_to_end or mention_level in config.json.")
			sys.exit(0)

		self.MODEL_OPTIONS = cf['model_options']

		self.TRAIN_FILENAME = check_filename_exists("data/datasets/%s/train.json" % cf['dataset'])
		self.DEV_FILENAME  = check_filename_exists("data/datasets/%s/dev.json" % cf['dataset'])
		self.TEST_FILENAME  = check_filename_exists("data/datasets/%s/test.json" % cf['dataset'])

		self.FILTERED_TRAIN_FOLDER = initialise_folder("data/datasets/%s_filtered" % cf['dataset'], "filtered dataset")
		self.FILTERED_TRAIN_VEC_FILENAME = self.FILTERED_TRAIN_FOLDER + "/train.vec"
		self.FILTERED_TRAIN_FILENAME = self.FILTERED_TRAIN_FOLDER + "/train.json"

		self.FILTERED_HIERARCHY_TRAIN_FOLDER = initialise_folder("data/datasets/%s_filtered_(hierarchy)" % cf['dataset'], "filtered dataset")
		self.FILTERED_HIERARCHY_TRAIN_FILENAME = self.FILTERED_HIERARCHY_TRAIN_FOLDER + "/train.json"

		self.MODEL_FOLDER 			= initialise_folder("models/%s_%s" % (cf['model'], options_to_text(self.MODEL_OPTIONS)), "model")
		self.MODEL_DATASET_FOLDER 	= initialise_folder("%s/%s_[%s train, %s test]" % (self.MODEL_FOLDER, cf['dataset'], self.MAX_SENTS["train"], self.MAX_SENTS["test"]), "model+dataset")
		self.DEBUG_FOLDER 			= initialise_folder("%s/debug" % (self.MODEL_DATASET_FOLDER), "asset")
		self.ASSET_FOLDER 			= initialise_folder("%s/asset" % (self.MODEL_DATASET_FOLDER), "asset")
		self.BEST_MODEL_FOLDER 			= initialise_folder("%s/best_model" % (self.MODEL_DATASET_FOLDER), "best model")
		self.BEST_MODEL_FILENAME		= "%s/model" % self.BEST_MODEL_FOLDER
		self.BEST_MODEL_JSON_FILENAME	= "%s/model.json" % self.BEST_MODEL_FOLDER

		self.EMBEDDING_DIM = cf['embedding_dim']
		self.HIDDEN_DIM = cf['hidden_dim']

		# Add the FileHandler to the logger if it doesn't already exist.
		# The logger will log everything to models/<model name>/log.txt.
		if len(logger.root.handlers) == 1:
			t =  datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
			d = initialise_folder("%s/log_%s" % (self.DEBUG_FOLDER, t), "log")
		
			for folder in [d, self.DEBUG_FOLDER]:
				log_filename = "%s/log.txt" % folder
				hdlr = logger.FileHandler(log_filename, 'w+')
				hdlr.setFormatter(LogToFile())
				logger.root.addHandler(hdlr)
				config_dump_filename = "%s/config.txt" % folder
				self.dump_config_to_file(t, config_dump_filename)



	# Dump all of this config object's field variables to the given file.
	# 't' is the current time, which is appended to the top of the file.
	def dump_config_to_file(self, t, fname):
		obj = self.__dict__
		with open(fname, 'w') as f:
			f.write("Config at %s\n" % t)
			f.write("=" * 80)
			f.write("\n")
			for items in sorted(obj.items()):
				f.write(": ".join([str(x) for x in items]))
				f.write("\n")
		logger.debug("Dumped config to %s." % fname)



# Load a config object, which is built using the config.json file.
def load_config():
	with open('config.json', 'r') as f:
		config_json = json.load(f)
		cf = Config(config_json)
	return cf
