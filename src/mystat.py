
##############################For vSRL ################################
def get_violated_verbs(res, margin):
	print "Before calibrating, %s verbs are not satisfied"%(res[abs(res['bef_diff']) > margin].count().verb)
	print "After calibrating, %s verbs are not satisfied"%(res[abs(res['after_diff']) > margin].count().verb)


def get_dis_gound_all(res, margin):
	bef_dis = abs(res['bef_diff']).sum()
	aft_dis = abs(res['after_diff']).sum()
	print "Before calibrating, distance to reference(training ratio) for all verbs:", bef_dis
	print "After calibrating, distance to reference(training ratio) for all verbs:", aft_dis
	print "Before calibrating, distance to reference(training ratio) for  violated verbs:", (abs(res[(res['bef_ratio'] > res['training_ratio'] + margin)]['bef_diff']).sum() +
		abs(res[(res['bef_ratio'] < res['training_ratio'] - margin)]['bef_diff']).sum())
	print "After calibrating, distance to reference(training ratio) for violated verbs:", (abs(res[(res['after_ratio'] > res['training_ratio'] + margin)]['after_diff']).sum() +
           abs(res[(res['after_ratio'] < res['training_ratio'] - margin)]['after_diff']).sum())

def dis_margin(res,margin):
	res['bef_upper_diff'] = res['bef_ratio'] - res['training_ratio'] - margin
	res['bef_lower_diff'] = res['bef_ratio'] - res['training_ratio'] + margin
	res['after_upper_diff'] = res['after_ratio'] - res['training_ratio'] - margin
	res['after_lower_diff'] = res['after_ratio'] - res['training_ratio'] + margin
	print "Before calibrating, the distance to the margin for the violated verbs", (res[(res['bef_ratio'] > res['training_ratio'] + margin)]['bef_upper_diff'].sum() +
       abs(res[(res['bef_ratio'] < res['training_ratio'] - margin)]['bef_lower_diff']).sum())
	print "After calibrating, the distance to the margin for the violated  verbs", (res[(res['after_ratio'] > res['training_ratio'] + margin)]['after_upper_diff'].sum() +
         abs(res[(res['after_ratio'] < res['training_ratio'] - margin)]['after_lower_diff']).sum())

def get_bias_score(res,margin):
	#calculate the bias score before LR
	woman_bias = (res[res['training_ratio'] <0.5]['training_ratio'] - res[res['training_ratio'] <0.5]['bef_ratio']).sum()
	man_bias = (res[res['training_ratio'] > 0.5]['bef_ratio'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
	print "Before calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(
		str((woman_bias + man_bias)/res.shape[0]))
	#calculate the bias score after LR
	woman_bias = (res[res['training_ratio'] <0.5]['training_ratio'] - res[res['training_ratio'] <0.5]['after_ratio']).sum()
	man_bias = (res[res['training_ratio'] > 0.5]['after_ratio'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
	print "After calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(
		str((woman_bias + man_bias)/res.shape[0]))


##############For coco ###############
def coco_get_violated_objs(df, margin):
    print "Before calibrating, %s objects are not satisfied"%(str(df[abs(df['bef_diff']) > margin].count().obj_id))
    print "After calibrating, %s objects are not satisfied"%(str(df[abs(df['aft_diff']) > margin].count().obj_id))

def coco_get_dis_ground(df, margin):
    print "Distance to reference(training ratio) for all verbs before calibrating: ", abs(df['bef_diff']).sum()
    print "Distance to reference(training ratio) for all verbs after calibrating: ", abs(df['aft_diff']).sum()

def coco_get_bias_score(df, margin):
    woman_bias = (df[df['train_ratio'] < 0.5]['train_ratio'] - df[df['train_ratio'] < 0.5]['bef_ratio']).sum()
    man_bias = (df[df['train_ratio'] > 0.5]['bef_ratio'] - df[df['train_ratio'] > 0.5]['train_ratio']).sum()
    print "Before calibrating, mean bias score for all objects = (woman_bias + man_bias)/#(objects): %s "%(str((woman_bias + man_bias) / df.shape[0]))

    woman_bias = (df[df['train_ratio'] < 0.5]['train_ratio'] - df[df['train_ratio'] < 0.5]['aft_ratio']).sum()
    man_bias = (df[df['train_ratio'] > 0.5]['aft_ratio'] - df[df['train_ratio'] > 0.5]['train_ratio']).sum()
    print "After calibrating, mean bias score for all objects = (woman_bias + man_bias)/#(objects): %s "%(str((woman_bias + man_bias) / df.shape[0]))
