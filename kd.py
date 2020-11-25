import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import ProbShift

from datalmdb import DataLmdb
from mfn import MfnModel
from mfn_mini import MfnModelMini


# Set device to be trained on

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define student and teacher models

teacher_model = MfnModel().cuda()
teacher_model.load_state_dict( torch.load('mfn_2510.pth') )
teacher_model.freeze()
student_model = MfnModelMini().cuda()

# Define optimizers

teacher_optimizer = optim.SGD(teacher_model.fc3_256_1.parameters(), lr=0.01)
student_optimizer = optim.SGD(student_model.parameters(), lr=0.01)


train_loader = torch.utils.data.DataLoader(DataLmdb("/kaggle/working/Low_Test/Train-Low_lmdb", db_size=143432, crop_size=128, flip=True, scale=0.00390625),
    batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(DataLmdb("/kaggle/working/Low_Test/Valid-Low_lmdb", db_size=7939, crop_size=128, flip=False, scale=0.00390625, random=False),
    batch_size=256, shuffle=False)

# Train using KD_Lib

distiller = ProbShift(teacher_model, student_model, train_loader, test_loader, teacher_optimizer,
                      student_optimizer, device=device)
distiller.train_teacher(epochs=5)                                       # Train the teacher model
distiller.train_student(epochs=5)                                      # Train the student model
distiller.evaluate(teacher=True)                                        # Evaluate the teacher model
distiller.evaluate()                                                    # Evaluate the student model