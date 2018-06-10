import os,sys

mskTxt = sys.argv[1]
tgtTxt	= sys.argv[2]

mskF = open(mskTxt,'w')
tgtF = open(tgtTxt,'w')
counter = 0
for i in range(1,11000):
	mskName = str(i)+'.glpOPC.png'
	tgtName = str(i)+'.glp.png'
	if os.path.isfile('../data/artimsk/'+mskName) and os.path.isfile('../data/artitgt/'+tgtName):
		mskF.write('../data/artimsk/'+mskName+'\n')
		tgtF.write('../data/artitgt/'+tgtName+'\n')
		counter+=1
	else:
		print(str(i)+'th pair doesn\'t exist')

print('total '+str(counter)+' training samples')
