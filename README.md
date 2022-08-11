***************************************************************
* CS302-Python-Group 38
* by Michelle Lu mlu415@aucklanduni.ac.nz 
* and Matthew Wu fwu277@aucklanduni.ac.nz
*THIS IS A REUPLOAD FROM AN EARLIER PROJECT
*
***************************************************************
RUNNING THE CODE:
1) Before you run any code you may need to install all the required 
libraries namely and are all incorporated in the interpreter;
	-Pytorch
	-Torch and Torchvision
	-Sklearn

2) To open the files in a Python IDE, double click the selected IDE to open
the interface, inside of the Python IDE, you can go File->Open Project and then
from there select to open the MINIST folder. Doing so will automatically
open all the other necessary files.

3) Running the code, There are four main algorithms with the normalize function 
available on all models;
To run Basic RNN
	-Ensure line 173 and 194 (data = torch.squeeze(data)) are uncommented
	-Ensure line 227 MLP_ is set to False
	-Ensure line 228 B_RNN is set to True
	-Ensure line 229 L_RNN is set to False
	-If input normalisation is desired check line 217 (Normalize = True)
	otherwise (Normalize = False)
To run LSTM RNN
	-Ensure line 173 and 194 (data = torch.squeeze(data)) are uncommented
	-Ensure line 227 MLP_ is set to False
	-Ensure line 228 B_RNN is set to False
	-Ensure line 229 L_RNN is set to True
	-If input normalisation is desired check line 217 (Normalize = True)
	otherwise (Normalize = False)
To run MLP
	-Ensure line 173 and 194 (data = torch.squeeze(data)) are uncommented
	-Ensure line 227 MLP_ is set to True
	-Ensure line 228 B_RNN is set to False
	-Ensure line 229 L_RNN is set to False
	-If input normalisation is desired check line 217 (Normalize = True)
	otherwise (Normalize = False)
To run LeNet CNN
	-Ensure line 173 and 194 (data = torch.squeeze(data)) are commented
	-Ensure line 227 MLP_ is set to False
	-Ensure line 228 B_RNN is set to False
	-Ensure line 229 L_RNN is set to False
	Setting all other models to False allows LeNet CNN to run
	-If input normalisation is desired check line 217 (Normalize = True)
	otherwise (Normalize = False)
4) To save model and display learning curve:	
	Ensure save_model = True 
