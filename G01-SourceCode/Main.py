from ProcessesCtrl import Process

InputDCMDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\DCM Files\\"

InputGTDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\GroundTruth\\"

OutputAVIDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\AVI Files\\"

OutputSkeletonizeDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\Skeleton Files\\"

OutputSegmentDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\Segment Files\\"

OutputResultDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\Result Files\\"

Process(InputDCMDir, InputGTDir, OutputAVIDir, OutputSkeletonizeDir, OutputSegmentDir, OutputResultDir)
