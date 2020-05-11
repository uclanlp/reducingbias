

def get_violated_verbs(res, margin):
    print ("[Inference]: \nBefore calibrating, %d verbs are not satisfied"%(res[abs(res['bef_diff']) > margin].count().verb_id))
    print ("After calibrating, %d verbs are not satisfied"%(res[abs(res['after_diff']) > margin].count().verb_id))
    print ("[Posterior probability]: \nBefore calibrating, %d verbs are not satisfied"%(res[abs(res['bef_diff_PR']) > margin].count().verb_id))
    print ("After calibrating, %d verbs are not satisfied"%(res[abs(res['after_diff_PR']) > margin].count().verb_id))
    return (res[abs(res['bef_diff']) > margin].count().verb_id, res[abs(res['after_diff']) > margin].count().verb_id, 
            res[abs(res['bef_diff_PR']) > margin].count().verb_id, res[abs(res['after_diff_PR']) > margin].count().verb_id)


def get_bias_score(res, margin):
    # calculate the bias score of inference results
    print ("[Inference results]")
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['bef_ratio']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['bef_ratio'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("Before calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))
    #calculate the bias score after LR
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['after_ratio']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['after_ratio'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("After calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))

    # calculate the bias score of distribution
    print ("[Posterior distribution]")
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['bef_ratio_PR']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['bef_ratio_PR'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("Before calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))
    #calculate the bias score after LR
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['after_ratio_PR']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['after_ratio_PR'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("After calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))