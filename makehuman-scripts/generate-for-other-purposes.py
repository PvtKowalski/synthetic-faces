# TODO: 
# 1. get values from config file
# 3. script only works after you do wierd things in editor

global RNG
RNG = G.app.getPlugin('0_modeling_10_random') # muh custom random plugin


def generateSpecificHuman(age, gender, race):
	"""age is a string 'young', 'middleage', 'old'
	gender is a string 'male', 'female'
	race is a string 'caucasian', 'african', 'asian'"""
	import numpy as np
	import os
	G.app._resetHuman()
	import skeleton
	defaultSkeleton = skeleton.load("data/rigs/default.mhskel")
	rand_weight = np.random.uniform(0.0, 0.8)
	if gender == 'male':
		rand_gender = 0.9 + np.random.uniform(0.0,0.1)
	elif gender == 'female':
		rand_gender = np.random.uniform(0.0,0.1)
	male_indicator = rand_gender > 0.5 # True - male
	
	# age from 0.5 to 1.0 where 0.5 is 25y 1.0 is 90y
	if age == 'young':
		rand_age = np.random.uniform(0.5, 0.6)
	elif age == 'middleage':
		rand_age = np.random.uniform(0.7, 0.8)
	elif age == 'old':
		rand_age = np.random.uniform(0.9, 1.0)
	
	rand_muscle = np.random.uniform(0.2, 0.7)
	rand_height = np.random.uniform(0.45, 0.8)
	rand_body_proportions = np.random.uniform(0.2, 0.7)
	
	rand_race = np.random.uniform(0.8, 1.0)
	# race_indicator = np.argmax(rand_race) # 0 - white, 1 - black, 2 - asian
	human = G.app.selectedHuman
	human.setSkeleton(defaultSkeleton)
	human.setWeight(rand_weight)
	human.setGender(rand_gender)
	human.setAge(rand_age)
	human.setBodyProportions(rand_body_proportions)
	human.setHeight(rand_height)
	human.setMuscle(rand_muscle)
	if race == 'caucasian':
		human.setCaucasian(rand_race)
	elif race == 'african':
		human.setAfrican(rand_race)
	elif race == 'asian':
		human.setAsian(rand_race)

	RNG.randomize(human, 0.5, False, False, True, True) # randomize face and body
	
	age_str = None
	race_str = race
	gender_str = None

	if male_indicator:
		gender_str = 'male'
	else:
		gender_str = 'female'

	# if race_indicator == 0:
		# race_str = 'caucasian'
	# elif race_indicator == 1:
		# race_str = 'african'
	# else:
		# race_str = 'asian'

	if rand_age <= 0.6:
		age_str = 'young'
	elif rand_age <= 0.8:
		age_str = 'middleage'
	else:
		age_str = 'old'

	# assign material
	mat = material.fromFile('data/skins/'+age_str +'_'+race_str+'_'+gender_str+'/'+age_str+'_'+race_str+'_'+gender_str+'.mhmat')
	human.material = mat

	# set teeth
	teeth_plugin = G.app.getPlugin('3_libraries_teeth')
	teeth_plugin.taskview.proxyFileSelected('data/teeth/teeth_base/teeth_base.mhpxy')

	# set tongue
	tongue_plugin = G.app.getPlugin('3_libraries_tongue')
	tongue_plugin.taskview.proxyFileSelected('data/tongue/tongue01/tongue01.mhpxy')

	# set ['eyebrows', 'eyelashes', 'hair']
	categories_list = ['eyebrows', 'eyelashes', 'hair', 'clothes', 'shoes'] 
	category_plugin_str = \
	{'eyebrows': '3_libraries_eyebrows',
	'eyelashes': '3_libraries_eyelashes',
	'hair': '3_libraries_polygon_hair_chooser', 
	'clothes': '3_libraries_clothes_chooser',
	'shoes': '3_libraries_clothes_chooser'}
	category_paths = \
	{'eyebrows': 'data/eyebrows/',
	'eyelashes': 'data/eyelashes/',
	'hair': 'data/hair/', 
	'clothes': 'data/clothes/',
	'shoes': 'data/clothes/shoes/'}
	for category in categories_list:
		category_plugin = G.app.getPlugin(category_plugin_str[category])
		category_path = category_paths[category]
		if category != 'shoes':
			if male_indicator:
				category_path = category_path + category + '_male/'
			else:
				category_path = category_path + category + '_female/'
		category_list = os.listdir(category_path)
		category_random = np.random.randint(0, len(category_list))
		resulting_path = category_path+category_list[category_random]+'/' + category_list[category_random]+'.mhpxy'
		category_plugin.taskview.proxyFileSelected(resulting_path)


#accessing export plugin
exportMhx2 = G.app.getPlugin('9_export_mhx2')
#setting up export config
cfg = exportMhx2.Mhx2Config()
cfg.human = G.app.selectedHuman
cfg.useTPose = False
cfg.useBinary = True # compression
cfg.useExpressions = True
cfg.usePoses = True
cfg.feetOnGround = True
cfg.scale = 1.0
cfg.unit = "m"


# How much and what to generate
how_much = 1 # generate of each type
ages = ['young', 'middleage', 'old']
races = ['caucasian', 'african', 'asian']
genders = ['male', 'female']

result_folder = 'models_cloth_50each'
result_path = 'C:/Users/пк1/Desktop/blender/exportScript/tst/'+result_folder+'/'

#get all info in file
f = open(result_path+'info.csv'.decode('utf-8'), 'w')
f.write(",filename,race,gender,age,age_years\n")
first_index = 0
global_index = 0
for a_race in races:
	for a_gender in genders:
		for an_age in ages:
			for i in range(how_much):
				generateSpecificHuman(an_age, a_gender, a_race)
				current_index = '{num:0{width}}'.format(num=first_index + global_index, width=5)
				exportMhx2.mh2mhx2.exportMhx2(result_path+'test_'+current_index+'.mhx2'.decode('utf-8'), cfg)
				human = G.app.selectedHuman
				f.write(str(first_index + global_index)+","+'test_'+current_index+'.mhx2'+","+a_race+","+a_gender+","+an_age+","+str(human.getAgeYears())+"\n")
				global_index += 1
f.close()




