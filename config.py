import argparse

def get_train_arguments():
    parser = argparse.ArgumentParser(description='train-poison-teacher-network')
    # Basic model parameters.
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--attack', type=str, default='patch', choices=['patch','blended', 'sig','bpp','wanet'])
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','preact_resnet'])
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--data', type=str, default='./cache/data')
    parser.add_argument('--output_dir', type=str, default='./cache/weights/')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--portion', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument("--lr_schedule", nargs="+", default=[100, 200, 300], type=int)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
    parser.add_argument("--mode", type=str, default='train', choices=['train'])
    parser.add_argument('--method', type=str, default='', choices=['CLP', 'NAD', 'FP', 'ANP','i-bau'])
    # for blended
    parser.add_argument('--weights', type=float, default=0.1)
    # for sig
    parser.add_argument('--delta', type=float, default=50)
    parser.add_argument('--f', type=int, default=6)
    # for wanet
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=4)
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--grid-rescale", type=float, default=1)
    # for bpp
    parser.add_argument("--n_iters", type=int, default=600)
    parser.add_argument("--neg_rate", type=float, default=0.1)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)
    return parser



# #for cifar10
# def get_defense_arguments():
#     parser = argparse.ArgumentParser(description='ADBR on CIFAR10')
#     # for Generator
#     parser.add_argument('--oh', type=float, default=1, help='one hot loss')
#     parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
#     parser.add_argument('--a', type=float, default=0.1, help='activation loss')
#     # basic
#     parser.add_argument("--seed", type=int, default=47)
#     parser.add_argument("--mode", type=str, default='defense', choices=['defense',"contrast"])
#     parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
#     parser.add_argument('--attack', type=str, default='patch', choices=['patch', 'wanet', 'blended', 'sig','bpp'])
#     parser.add_argument('--target', type=int, default=0)
#     parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','preact_resnet'])
#     parser.add_argument('--portion', type=float, default=0.1)
#     parser.add_argument('--data', type=str, default='./cache/data')
#     parser.add_argument('--output_dir', type=str, default='./cache/weights/')
#     parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
#     parser.add_argument('--epochs', type=int, default=400, help='number of epochs of training')
#     parser.add_argument('--GS_iters', type=int, default=5, help='number of epochs of Student and Generator training')
#     parser.add_argument('--S_iters', type=int, default=5, help='number of epochs of Student training')
#     parser.add_argument('--channels', type=int, default=3, help='number of image channels')
#     parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
#     parser.add_argument('--lr_decay', type=float, default=0.5)
#     parser.add_argument("--lr_schedule", nargs="+", default=[200], type=int)
#     parser.add_argument('--method', type=str, default='')

#     #patch=4 blended=2 sig=2 bpp=3 wanet=4
#     parser.add_argument('--shuffle_layers', type=int, default=4)
#     #patch=4 blended=4 sig=3 bpp=3 wanet=4
#     parser.add_argument('--tea_shuffle_layers', type=int, default=4)
#     parser.add_argument('--n_shuf_ens', type=int, default=3)
#     parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
#     parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
#     parser.add_argument('--lr_G', type=float, default=1e-3, help='Generator learning rate')
#     parser.add_argument('--lr_S', type=float, default=2e-3, help='Student learning rate')
#     parser.add_argument('--lamda', type=int, default=0.01)
#     parser.add_argument('--alpha', type=int, default=0.02)

#     # for blended 
#     parser.add_argument('--weights', type=float, default=0.1)
#     # for sig 
#     parser.add_argument('--delta', type=float, default=30) 
#     parser.add_argument('--f', type=int, default=6)
#     # for wanet
#     parser.add_argument("--random_crop", type=int, default=4)
#     parser.add_argument("--cross_ratio", type=float, default=2)
#     parser.add_argument("--k", type=int, default=4)
#     parser.add_argument("--s", type=float, default=0.5)
#     parser.add_argument("--grid-rescale", type=float, default=1)
#     # for bpp
#     parser.add_argument("--neg_rate", type=float, default=0.2)
#     parser.add_argument("--squeeze_num", type=int, default=8)
#     parser.add_argument("--dithering", type=bool, default=False)
#     return parser


#for gtsrb
def get_defense_arguments():
    parser = argparse.ArgumentParser(description='ADBR on GTSRB')
    # for Generator
    parser.add_argument('--oh', type=float, default=1, help='one hot loss')
    parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
    parser.add_argument('--a', type=float, default=0.1, help='activation loss')
    # basic
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--mode", type=str, default='defense', choices=['defense',"contrast"])
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--attack', type=str, default='patch', choices=['patch', 'wanet', 'blended', 'sig','bpp'])
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','preact_resnet'])
    parser.add_argument('--portion', type=float, default=0.1)
    parser.add_argument('--data', type=str, default='./cache/data')
    parser.add_argument('--output_dir', type=str, default='./cache/weights/')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs of training')
    parser.add_argument('--GS_iters', type=int, default=5, help='number of epochs of Student and Generator training')
    parser.add_argument('--S_iters', type=int, default=5, help='number of epochs of Student training')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument("--lr_schedule", nargs="+", default=[200], type=int)
    parser.add_argument('--method', type=str, default='')

    parser.add_argument('--shuffle_layers', type=int, default=2)#sig=4 others=2
    parser.add_argument('--tea_shuffle_layers', type=int, default=3)
    parser.add_argument('--n_shuf_ens', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--lr_G', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--lr_S', type=float, default=3e-3, help='Student learning rate')
    parser.add_argument('--lamda', type=int, default=0.01)
    parser.add_argument('--alpha', type=int, default=0.02)

    # for blended 
    parser.add_argument('--weights', type=float, default=0.1)
    # for sig 
    parser.add_argument('--delta', type=float, default=50)
    parser.add_argument('--f', type=int, default=6)
    # for wanet
    parser.add_argument("--random_crop", type=int, default=4)
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--grid-rescale", type=float, default=1)
    # for bpp
    parser.add_argument("--neg_rate", type=float, default=0.2)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)
    return parser


