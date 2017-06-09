# evaluation-function
 
**Python Code**
 
* name：cross.validation.stable.py

* default parame

* Need to modify the parameter：`-`_dataSrc
 	* The path is data source`-`columsNum:	* the numbers of columns quantity`-`_ output_folderName
	* The name is for data output path`-` num_class	* The numbers of classify amount**Parameter Description**

* num_round
```	The number of rounds for boosting
```

* eta [default=0.3]

```
	Analogous to learning rate in GBM
		Makes the model more robust by shrinking the weights on each step
```
* max_depth[default=6]```	The maximum depth of a tree, same as GBM.	Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.	Should be tuned using CV.```