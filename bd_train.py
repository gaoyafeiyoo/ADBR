from torch.autograd import Variable

from utils.wanet import *
from utils.bpp import *
from utils.get_model_loader import *
from utils.utils import more_config,set_random_seed,adjust_learning_rate
from config import get_train_arguments

args = get_train_arguments().parse_args()
more_config(args)
os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0
asr_best = 0

data_test = get_data_test(args)
train_loader,test_loader_clean,test_loader_dirty = get_train_loader(args)
net = get_model(args)

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()
        output,*_ = net(images)
        loss = criterion(output, labels)
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
        if i == 1:
            print('Train - Epoch %d, Loss: %.4f' % (epoch, loss.data.item()))
        loss.backward()
        optimizer.step()
 
 
def test(test_loader):
    global acc 
    net.eval()
    total_correct = 0
    total_error = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output,*_ = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            results = pred.eq(labels.data.view_as(pred))
            total_correct += results.sum()
            total_error += (~results).sum()
 
    avg_loss /= len(data_test)
    acc = total_correct * 100 / (total_correct + total_error)
    to_log_file('Test Avg. Loss: %.3f, Accuracy: %.2f' % (avg_loss.data.item(), acc)+'\n',args.checkpoint, 'train_log.txt')
    return acc


def train_and_test(epoch):
    adjust_learning_rate(optimizer, epoch,args.lr_schedule, args.lr_factor)
    global acc_best,asr_best
    if args.attack == 'wanet':
        noise_grid, identity_grid = prepare(args)
        train_wanet(args, net, train_loader, criterion, optimizer, noise_grid, identity_grid, epoch)
        acc, asr, acc_cross, _ = test_wanet(args, net, test_loader_clean, noise_grid, identity_grid)
        print()
        to_log_file(
        'Clean acc %.2f BD asr %.2f Cross acc %.2f ' % (acc, asr, acc_cross)+'\n',
        args.checkpoint, 'train_log.txt')
        state = {
        'state_dict': net.state_dict(),
        "identity_grid": identity_grid,
        "noise_grid": noise_grid,
        }
        if acc - 0.03 > acc_best or acc > acc_best - 0.03 and asr > asr_best :
            acc_best, asr_best = acc, asr
            to_log_file("Saving Best model...",args.checkpoint, 'train_log.txt')
            model_save(args,state=state)
            
    elif args.attack == 'bpp':
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0

        residual_list_train = []
        count = 0
        for j in range(5):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                temp_negetive = back_to_np_4d(inputs, args)
                temp_negetive_modified = back_to_np_4d(inputs, args)

                if args.dithering:
                    for i in range(temp_negetive_modified.shape[0]):
                        temp_negetive_modified[i, :, :, :] = torch.round(torch.from_numpy(
                            floydDitherspeed(temp_negetive_modified[i].detach().cpu().numpy(), float(args.squeeze_num))))
                else:
                    temp_negetive_modified = torch.round(temp_negetive_modified / 255.0 * (args.squeeze_num - 1)) / (
                                args.squeeze_num - 1) * 255

                residual = temp_negetive_modified - temp_negetive

                for i in range(residual.shape[0]):
                    residual_list_train.append(residual[i].unsqueeze(0).cuda())
                    count = count + 1

        residual_list_test = []
        count = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader_clean):
            temp_negetive = back_to_np_4d(inputs, args)
            residual = torch.round(temp_negetive / 255.0 * (args.squeeze_num - 1)) / (
                        args.squeeze_num - 1) * 255 - temp_negetive
            for i in range(residual.shape[0]):
                residual_list_test.append(residual[i].unsqueeze(0).cuda())
                count = count + 1

        for epoch in range(0, args.n_iters):
            print("Epoch {}:".format(epoch + 1))
            train_bpp(args, net, optimizer, train_loader,residual_list_train)
            clean_acc, bd_acc, cross_acc = test_bpp(args, net, test_loader_clean, residual_list_test)

            if clean_acc > best_clean_acc or (clean_acc > best_clean_acc - 1 and bd_acc > best_bd_acc):
                print(" Saving...")
                best_clean_acc = clean_acc
                best_bd_acc = bd_acc
                if args.neg_rate:
                    best_cross_acc = cross_acc
                else:
                    best_cross_acc = torch.tensor([0])
                to_log_file(
                    "Best Clean Acc: {:.4f} | Best Asr: {:.4f} | Best Cross: {:.4f} ".format(
                    best_clean_acc, best_bd_acc, best_cross_acc) + '\n',
                    args.checkpoint, 'train_log.txt')
                    
                state_dict = {
                    "state_dict": net.state_dict(),
                }
                to_log_file("Saving Best model...", args.checkpoint, 'train_log.txt')
                model_save(args, state=state_dict)
    else:
        train(epoch)
        acc = test(test_loader_clean)
        asr = test(test_loader_dirty)
        # if acc > 70:
        #     if (acc - 3 > acc_best or acc > acc_best - 3) and asr > asr_best :
        #         acc_best, asr_best = acc, asr
        #         to_log_file("Saving Best model...",args.checkpoint, 'train_log.txt')
        #         model_save(args,net)

        if acc > acc_best and asr > 95:
            acc_best, asr_best = acc, asr
            to_log_file("Saving Best model...",args.checkpoint, 'train_log.txt')
            state_dict = {
                    "state_dict": net.state_dict(),
                }
            to_log_file("Saving Best model...", args.checkpoint, 'train_log.txt')
            model_save(args, state=state_dict)

def main(args):
    set_random_seed(args)
    if args.attack in ['bpp']:
        args.epochs = 2
    for i in range(1, args.epochs):
        train_and_test(i)

if __name__ == '__main__':
    main(args)