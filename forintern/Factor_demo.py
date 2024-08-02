from Factor import factor_evaluate_name_pred,, factor_evaluate_num


name_list += ['factor/factor1.csv']
p = Pool(min(len(name_list),100))
#p.map(factor_evaluate_num, name_list)
p.map(factor_evaluate_name_pred, name_list)
