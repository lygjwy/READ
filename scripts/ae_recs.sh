TEST_DATASETS='isun lsunc lsunr places365 tinc tinr dtd'
DUAL_DATASETS='cifar10 cifar100 svhn'

for test_dataset in $TEST_DATASETS; do
python ae_rec.py --data_dir /home/iip/datasets --dataset $test_dataset --split test --data_mode original --batch_size 128 --prefetch 4 --arch resnet18 --ae_path ./assets/rec_2000_best.pth --gpu_idx 0
python ae_rec.py --data_dir /home/iip/datasets --dataset $test_dataset --split test --data_mode corrupt --batch_size 128 --prefetch 4 --arch resnet18 --ae_path ./assets/rec_2000_best.pth --gpu_idx 1
python /home/iip/utils/make_dataset.py --root /home/iip/datasets --datasets ${test_dataset}{or,cor}
done

for dual_dataset in $DUAL_DATASETS; do
python ae_rec.py --data_dir /home/iip/datasets --dataset $dual_dataset --split train --data_mode original --batch_size 128 --prefetch 4 --arch resnet18 --ae_path ./assets/rec_2000_best.pth --gpu_idx 2
python ae_rec.py --data_dir /home/iip/datasets --dataset $dual_dataset --split train --data_mode corrupt --batch_size 128 --prefetch 4 --arch resnet18 --ae_path ./assets/rec_2000_best.pth --gpu_idx 1
python ae_rec.py --data_dir /home/iip/datasets --dataset $dual_dataset --split test --data_mode original --batch_size 128 --prefetch 4 --arch resnet18 --ae_path ./assets/rec_2000_best.pth --gpu_idx 0
python ae_rec.py --data_dir /home/iip/datasets --dataset $dual_dataset --split test --data_mode corrupt --batch_size 128 --prefetch 4 --arch resnet18 --ae_path ./assets/rec_2000_best.pth --gpu_idx 2
python /home/iip/utils/make_dataset.py --root /home/iip/datasets --datasets ${dual_dataset}{or,cor}
done

#  make datasets
