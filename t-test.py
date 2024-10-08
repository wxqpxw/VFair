import sys
import numpy as np
from scipy import stats

#
while True:
    try:
        mean_VFair, std_VFair = map(float, input("input mean_VFair and std_VFair，split by space:").split())
        mean_ERM, std_ERM = map(float, input("input mean_ERM and std_ERM，split by space:").split())

        n1, n2 = 10, 10
        t_stat_ind, p_val_ind = stats.ttest_ind_from_stats(mean_VFair, std_VFair, n1, mean_ERM, std_ERM, n2)

        print(f't = {t_stat_ind}, p = {p_val_ind}')
    except ValueError:
        print("value error")
    except KeyboardInterrupt:
        print("\nthe end")
        break

