from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset

# if __name__ == '__main__':
#     f = open('config.yml', 'r')
#     config = yaml.load(f, Loader=yaml.FullLoader)
#     print(config)


# if __name__ == '__main__':
#     print(cfg)
#     cfg['a'] = 123456
#     print(cfg)

if __name__ == '__main__':
    data_name = 'CelebA-HQ'
    cfg['subset'] = 'attr'
    cfg['batch_size']['train'] = 10
    dataset = fetch_dataset(data_name, cfg['subset'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset)
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        input_size = input['img'].size(0)
        print(input['img'].size(), input[cfg['subset']].size())
        exit()
