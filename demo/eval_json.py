import os
from tools.cityPerson.eval_demo import validate


if __name__ == "__main__":
    result_root = '/home/wiser-renjie/projects/blockcopy/Pedestron/results'
    exp_id = 'csp_blockcopy_t030'
    json_path = os.path.join(result_root, exp_id + '.json')

    MRs = validate('/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/val_gt.json', json_path)
    
    print('Checkpoint %d: [Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]'
            % (1, MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))