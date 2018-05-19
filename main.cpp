#include<iostream>
#include<fstream>
#include<string>
#include"ndarray.h"
#include"cifar10_reader.hpp"
#include"Layers.hpp"
using namespace std;

int main() {
	//auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

	double w1[64*1*3*3] = {
		0.173687,0.112629,-0.325587,1.63695,1.24262,0.114021,0.490452,-0.194724,-0.555919,0.028653,0.0555931,0.0447632,0.172394,0.0519527,-0.103215,-0.118575,0.0212875,0.115473,0.131696,0.361543,0.194398,-0.164957,0.382632,-0.0876038,-0.103722,-0.0523858,-0.0128118,0.000193747,-0.192429,0.0759708,0.270022,0.254222,0.137718,0.306729,0.188332,0.0635258,-0.154242,-0.171904,-0.118717,0.053621,0.44168,0.226827,-0.182248,-0.0519043,-0.0928149,-0.143477,0.282359,0.171388,-0.229716,0.355072,0.824329,-0.00049417,-0.0961535,-0.000281116,-0.0231738,-0.228464,-0.20912,0.203061,-0.0503037,-0.221337,-0.0579704,0.298456,0.192443,-0.112561,-0.0871174,0.177892,0.147069,-0.0727854,-0.151685,0.186062,-0.0564921,-0.153554,-0.0466525,-0.149654,-0.12878,-0.214386,0.140925,0.337531,-0.217174,0.118852,0.191637,-0.213221,-0.349036,-0.124197,-0.133062,0.187014,0.0858958,-0.264304,0.588803,0.231148,-0.0149207,0.00306457,0.0761835,-0.0848642,0.445401,-0.166403,0.0192014,-0.0501343,0.17331,-0.164499,-0.121115,0.33365,0.741416,1.06687,0.678131,-0.0662819,-0.538852,-0.49668,-0.138669,0.0960342,-0.103579,-0.208563,0.179445,0.0873053,-0.104194,0.56297,0.120259,-0.0608163,-0.0138259,0.0523295,-0.107063,-0.0560566,0.198108,-0.0763936,0.0263096,0.267612,-0.0731143,0.576845,-0.0944815,-0.0898301,0.415595,-0.14838,-0.385823,-0.195902,-0.25129,0.112906,0.296119,0.169499,0.180632,0.206119,0.0173488,-0.0247943,-0.0867273,-0.0527337,-0.236241,0.110149,0.149015,-0.232542,0.371686,0.0153617,-0.277904,-0.109281,-0.0485529,0.0741373,-0.0924574,0.24035,0.181567,-0.120566,0.233242,-0.231142,-0.0646008,-0.0181173,0.776572,-0.14043,0.309384,0.381626,-0.344403,0.055321,-0.0445226,-0.453195,-0.0987215,-0.042358,-0.061105,0.0394173,-0.122206,-0.090648,0.215714,-0.0644182,-0.0628733,-0.0419576,0.0339037,0.0671353,0.00580972,0.210627,0.551394,0.0129147,0.368052,0.315907,-0.0865159,0.0634797,-0.0413268,-0.0940613,0.191015,-0.0900402,0.101369,-0.00807357,-0.114245,0.0039215,-0.307717,0.0360824,0.0626087,-0.166866,0.140806,0.481883,-0.0437426,0.309578,0.0613127,-0.0848805,0.13662,0.104181,-0.149871,-0.256742,-0.0649682,0.12622,-0.106397,-0.10039,-0.0557531,0.215809,0.140029,0.101183,0.0632073,-0.0183419,-0.0634002,-0.24875,-0.161032,-0.105452,-0.288285,-0.15971,0.0126566,-0.0504189,-0.0706655,0.245461,0.281488,0.088813,0.240154,-0.131618,-0.0712838,0.110256,-0.0843669,-0.0688781,0.00527046,-0.0946162,-0.0387087,-0.0476528,-0.17104,-0.0773936,0.157102,0.052312,0.0470587,0.0111115,0.158671,0.168546,0.160106,0.141926,0.475789,0.288836,0.189314,0.278159,-0.257398,-0.15795,-0.249441,-0.145362,-0.10504,0.0887473,0.0980819,0.260661,0.0443626,0.0552462,-0.0190374,-0.0847097,0.000793365,0.236953,-0.132143,-0.0505833,-0.0853306,-0.121215,0.0904187,0.20675,0.0972077,-0.116349,-0.162217,-0.218429,0.89604,0.996311,0.804997,0.525151,0.679975,0.389926,0.26006,0.238678,-0.0939244,-0.0566554,-0.310968,-0.243033,0.033766,-0.215327,-0.299583,-0.139012,-0.0741317,0.305336,-0.133074,-0.0735424,0.183066,-0.0999165,-0.0636359,-0.0761814,-0.0511654,-0.0963811,0.0896784,-0.222703,-0.153037,0.317492,0.069635,-0.246791,-0.0125421,0.0362083,-0.109324,-0.113658,0.130345,-0.0623725,-0.128806,-0.0751019,0.557715,-0.101436,0.00813134,-0.104917,0.106438,-0.214529,-0.10576,0.222814,-0.26775,0.0706471,0.30032,-0.187433,-0.0543108,-0.090882,-0.0310688,0.0639551,0.194013,-0.0669893,0.405687,0.461664,-0.083443,0.155144,-0.0845329,0.16336,-0.0377347,-0.126138,0.0556308,-0.128521,-0.0553753,-0.0972512,0.243048,-0.0427076,-0.317835,0.0732384,0.317804,-0.31568,0.011903,0.487058,-0.30641,0.00291506,-0.0183589,0.173175,-0.173325,0.177682,0.419655,-0.275762,0.260283,0.2258,0.0277191,-0.233663,0.135096,-0.114237,-0.278597,0.0250419,0.135009,-0.257291,-0.103377,-0.0517679,-0.0832978,0.140554,0.106868,-0.0283918,0.461279,-0.0598385,-0.0665196,-0.0759476,-0.101421,-0.407649,0.152303,0.453682,0.185577,-0.101559,0.151144,0.266737,-0.221729,-0.0982077,-0.020337,0.347937,-0.0257168,-0.0991439,0.124797,0.0759394,-0.0473006,-0.0193284,-0.125944,-0.172786,-0.111747,0.180062,-0.0247108,-0.0313493,0.176855,0.349469,0.350838,-0.00961847,-0.203192,0.10966,-0.0443527,0.0185258,-0.15315,-0.142747,-0.0528799,-0.00490644,0.0249454,-0.178278,0.0929478,0.148403,0.005525,0.0554311,0.245676,0.115271,0.396425,0.0295437,-0.153746,0.864084,0.344755,0.129012,0.148733,0.18391,-0.0530789,-0.0518441,-0.142951,-0.0729659,0.072531,-0.0445689,-0.111141,-0.160966,0.233967,0.0708103,0.35338,0.463593,0.65206,0.080042,0.467993,0.665724,-0.0290623,0.229105,0.70323,0.0360805,-0.0756442,0.173634,0.230816,0.373821,0.116501,0.111668,0.203327,0.138413,0.0789035,0.279123,-0.171688,0.114268,0.145615,0.0077216,-0.315402,-0.149125,-0.0852839,0.399731,0.248968,0.584565,-0.0243516,-0.261178,0.191248,-0.22768,-0.402072,0.0472195,0.152823,-0.255539,0.0660875,-0.327361,-0.162177,-0.211436,-0.257179,-0.154127,-0.24844,-0.00355206,0.424606,0.550076,-0.0768258,-0.0939779,-0.0104892,-0.449571,-0.307905,-0.299958,0.784273,0.454967,-0.210379,0.755559,0.174071,-0.273648,0.699142,0.0403225,-0.136674,0.201828,-0.11852,-0.0911663,-0.0246533,-0.12897,-0.0872748,-0.0369407,-0.0937439,-0.0269676,-0.0515277,0.233652,0.173626,-0.189241,-0.212295,-0.0847548,0.0495526,-0.115729,-0.034594,0.13927,-0.0335276,-0.0202532,-0.210265,-0.0717698,0.330521,-0.226305,0.0318421,0.274357,0.205024,0.0714852,-0.273248,0.315171,0.202343,-0.323696,0.0538389,-0.112495,-0.0813072,-0.0836477,-0.127224,-0.0466001,-0.053057,-0.025541,0.231918,0.03126,0.174395,0.063461,-0.105963,-0.0375772,0.233889,-0.106411,-0.08935,0.100803,-0.0903126,-0.0657113,-0.0173511,-0.169368,-0.224448,0.0287198,-0.19498,-0.0363523,0.0914683,-0.0765768,-0.106914,0.394096


	};
	double input[64 * 8*8] = {
		2.19141,0.799593,0.995594,1.60939,2.33816,1.99172,0.524973,1.73762,1.34078,0.180203,0.285463,1.21334,0.0,0.960334,0.0,0.495033,1.97775,0.87286,0.0,0.0,0.0,1.97986,0.0,0.446454,0.433605,0.0,0.0,0.0,0.0,0.0,0.62069,0.625915,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.728367,0.0,0.0,1.08485,0.0,0.0,0.0,0.0,1.77446,0.0,0.0,0.0,0.0,0.0868141,0.294202,0.0,2.30866,0.0,0.0,0.0,0.19429,0.491506,0.146298,0.0,0.0,0.0283528,0.451949,0.766438,0.972176,0.427928,0.329329,0.515254,1.21005,0.0333639,0.570186,0.992247,1.08026,0.882456,0.0,0.0,1.07385,0.710176,1.70473,1.29703,0.751842,0.493317,0.0,0.264061,1.07368,1.08659,1.22178,0.420264,0.0331373,0.0,0.0,0.0,1.09915,1.39558,0.721429,0.315959,0.289463,0.0,0.0,0.0,1.0577,0.954239,0.0397124,0.0,0.0,0.401168,0.690361,0.0,0.0,0.978656,0.0,0.0,1.33704,0.976925,0.144043,0.0,0.0,1.00036,0.362232,0.0,0.908059,1.33844,0.0,0.0,0.0,0.554277,0.0,0.0,0.0,0.321001,0.0,0.314399,0.236185,0.783635,0.0,0.0,0.0,0.0,0.776979,1.35032,0.414485,0.712857,0.0,0.0,0.0,0.0,1.28742,0.876088,0.448255,1.10155,0.0,0.0,0.0,0.0,0.0,0.842397,0.492965,0.0,0.0,0.0,0.0,0.0,0.0,0.952247,0.56214,0.0,0.0,0.0,1.35277,0.0,0.102495,0.0,1.06712,0.0,0.0,0.646327,0.987929,0.0,0.0,0.0,0.218901,0.189508,0.0,0.498357,0.0,0.0,0.931393,0.0,1.22523,0.322982,0.244641,0.643644,0.771677,0.0,0.0249185,0.0,0.337962,0.10277,0.181732,0.304489,0.0,0.0,1.31244,0.0,0.0319753,0.0240094,0.0,0.0,0.460434,0.126664,0.0459726,0.0,0.0343881,0.0,0.0,0.0,0.247932,0.0,1.17472,1.2253,0.0666024,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.124995,0.0,0.252646,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.15361,0.0,0.0,0.186577,0.0,0.126851,3.52849,0.0,0.0860065,0.0292475,0.129457,0.0,0.386819,0.705607,1.54791,0.0,1.56474,2.1367,1.92821,0.0,0.0,0.815698,1.50805,0.0,1.26019,1.71833,0.0,1.19621,0.0,0.0,0.769888,0.0,0.0,0.0,0.0,0.405036,0.0,0.0,0.798192,0.509199,0.405807,0.132185,0.0329605,0.170457,0.26405,0.0,0.493825,0.141393,0.0,0.30652,0.77004,0.0,0.0,0.0,0.0,0.359262,0.414092,0.0,0.698573,0.28648,0.659548,0.0,0.0,0.541943,0.677866,0.0,1.15652,1.15363,0.0,0.0,0.0,0.0,0.683956,0.0,0.816791,1.25414,0.0,0.741071,0.339586,0.0,0.267135,0.0400402,0.0,0.0,0.0,0.0,0.156178,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.263639,0.0,0.0,0.0,0.0,0.909104,0.729733,0.239135,0.246963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.259599,0.0,0.0,0.112758,0.0,0.487241,1.18697,2.25265,0.0,0.0,0.0,0.0,0.263329,0.0,0.0,3.15457,2.32991,0.126313,0.0,0.0,1.20808,0.962121,0.603751,0.0,0.92418,0.812876,0.0,0.0,0.410178,0.709912,0.97043,0.0351083,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.762202,0.0,0.0,1.22333,0.0,0.0,0.0,0.279579,0.911445,1.57815,1.34592,0.702145,0.277523,0.0568991,0.0711482,0.291543,0.602698,1.19989,0.765432,0.718506,0.0,0.0,0.0,0.068969,2.21313,1.08774,0.0,0.0,1.38321,0.986705,0.148187,0.0,2.76995,0.0,0.0,0.0,0.0,0.253791,0.345617,0.0,2.13311,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.61517,0.0,0.0813224,0.248024,0.123532,0.0,0.0,0.0,0.755902,0.948053,0.0,0.0,0.0,0.299131,0.675041,0.134308,0.561809,0.0,0.0,0.0,0.0,0.0,0.361129,0.226385,0.0,0.0,0.0,0.0107004,0.474013,0.0,0.394417,0.222625,0.0,0.0,0.816734,1.09716,0.0,0.0,0.0,0.0,0.0,0.643098,0.631902,0.616217,1.16252,0.110693,0.0,0.0,0.0,0.645447,0.0,0.0,0.558745,0.161665,0.0,0.0,0.0,0.136021,0.0,0.0,0.0,0.0,0.0,0.22935,0.0,0.379353,1.25493,0.673995,0.0648181,0.0,0.0,0.0,0.359091,0.787786,1.03956,1.09862,0.640695,0.0,0.0,1.35611,1.19668,1.29553,0.0,0.0,0.0,0.0,0.659798,1.98255,1.15391,0.0,0.0,0.0,0.321507,0.0414044,1.39946,1.94678,0.0,0.0,0.830155,0.966977,0.0,0.0,1.38797,2.11137,0.0,0.233471,0.279503,0.0,0.597657,0.46028,0.0,1.5813,0.0,0.492437,0.0,0.0,0.797648,0.0,0.0,0.299753,0.0,0.0,0.0,0.34946,0.179915,0.0,0.0887736,0.377648,0.0,0.0,0.141459,1.06355,0.773116,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0428338,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.272174,0.0,0.0,0.0,0.203325,0.532903,0.0,0.0,0.0,0.00978693,0.0,0.0,0.393052,0.088038,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.192654,0.0,0.505475,0.0,0.0229528,0.0,0.0,0.0,0.114398,0.0141115,0.0,0.088086,0.0012336,0.0,0.33753,0.0,0.0,0.0,0.0,0.0,0.112327,0.406268,0.420015,1.02188,1.37842,0.713353,0.0,1.60861,0.00257959,1.47837,1.23565,0.416834,0.0,0.0,0.34924,1.16313,0.0,1.37714,0.781828,0.474258,1.84119,1.75379,0.329658,0.752936,0.0,0.19501,0.818523,1.58493,2.80232,1.09515,0.498748,0.0,0.0,0.0,0.592033,2.30112,1.33892,0.65837,1.20519,0.177854,0.575438,0.0,0.0,0.0,0.728539,0.0,1.13513,0.517319,1.67594,0.0,0.0,0.0,0.0397885,1.47683,0.183494,0.225278,0.0957695,0.956814,0.0,0.0,0.895444,0.901244,0.788378,0.71687,0.0,0.442102,0.883169,0.819147,0.918267,0.694001,0.378384,1.0443,1.04202,1.20816,0.519171,0.0,0.0404038,1.4167,2.54914,1.06393,0.0,0.517991,0.0,0.0,2.2615,2.65417,1.40551,0.453245,0.0,0.00506915,0.0,1.3093,1.23789,0.178501,0.0,0.0,0.493416,0.0,0.0,0.83081,0.300711,0.323898,0.495032,0.0,0.209523,0.0,0.0,0.0,0.296428,0.180408,1.35127,0.0,0.88497,0.0,0.0,0.0,0.0156086,0.982966,0.414424,0.583757,0.44118,0.736361,0.0,0.0,0.650981,0.656563,0.375788,0.409456,0.606147,1.08304,0.601273,1.35181,0.0,0.0,0.349811,0.0,0.912488,1.1817,0.0,0.623317,0.0,0.8125,3.6561,0.0,0.529338,0.504644,0.0,0.09763,2.75311,3.13157,1.92466,0.119266,0.0,0.71405,0.0,0.0300703,0.749721,0.0,0.0,0.0,0.0,0.0532751,0.0,0.872094,0.206852,0.0209986,0.0,1.00273,0.0,0.0,0.133276,0.0,0.0,0.793013,0.151672,0.0,0.265912,0.0,0.0,0.0,0.0,2.05357,0.0,0.0,0.700672,0.0,0.0,0.67696,0.178784,0.619061,0.0,0.0,0.0,0.544957,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0364427,0.0,0.0,0.0627804,0.0,0.252184,0.447897,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.426942,0.0,0.0,0.0,0.00951269,0.0,0.0,0.0,0.0,0.0,0.456114,0.0,0.0,0.0,0.0,0.0,0.276556,0.0,0.0,0.0,0.872433,0.47926,0.0,0.0,0.535,0.338488,0.206372,0.219618,0.0,0.0,0.0,0.0,0.434361,0.0,0.0,0.463278,0.0,0.0,0.0,0.0,0.809113,0.357707,0.0,0.0,0.0,0.0,0.0,0.0,0.223054,0.226481,0.0,0.0,0.00418241,0.0,0.0,0.0,0.0,1.31425,0.0,0.35184,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.30368,0.167126,0.683835,0.0,0.0,0.0,0.0,0.0,1.28217,1.1572,0.923727,0.0,0.0,0.0,0.0,1.86897,0.533282,0.0,0.022088,0.0,0.0,0.0,0.0,0.87782,0.44854,0.474385,0.500657,0.502928,0.154643,0.12927,0.377353,0.490595,0.309931,0.585991,0.576462,0.172617,0.114578,0.301233,0.250472,0.419452,0.544523,0.0,0.0,0.0,0.15498,1.1622,0.44171,0.41349,0.261063,0.0,0.16021,0.0152762,0.267757,1.46127,0.0172287,0.422206,0.0,0.0,0.0,0.0,0.0,0.871624,1.36807,0.57681,0.0,0.0,0.0,0.256307,0.0,0.226927,1.09059,1.97635,0.0,0.0,0.0,0.164455,0.315045,0.401381,0.0,1.08542,0.0,0.231325,0.35448,0.352816,0.364388,0.500684,0.735716,0.287627,0.779394,0.0,0.0,0.0,0.0,1.14055,0.0,0.0,0.19882,0.0,0.0,0.0,0.0,1.7267,0.0,0.0,0.0,0.0,0.0,0.0915062,0.20838,2.28621,0.0,0.0,0.0,0.0,0.0,0.0,2.72799,0.771186,0.782068,0.0,0.0,0.224482,0.413274,0.241305,0.0949603,0.0,2.39805,0.0,0.658174,0.0,2.74019,0.0,0.0296256,0.0,1.59017,0.624865,0.961846,1.35359,1.51967,0.0,0.557632,1.01745,0.162109,1.52803,0.247836,0.7612,0.0,0.0,0.0,0.416861,1.53895,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.822,0.0,0.0,0.0,0.0,0.0,2.27369,1.67831,0.764204,0.235856,0.0,0.0,0.0,0.0,2.45138,0.0,0.235048,0.0248411,0.0,0.0,0.0,2.50525,0.861936,0.0495633,0.0,0.837952,0.0,0.231009,0.0,3.34346,0.0,0.0,0.0,0.0,0.0,1.02874,0.0,2.61987,0.0,0.0,0.0696089,0.0,0.0,0.0,0.0,1.27167,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.517081,0.0,0.0,0.0,0.0,0.085207,0.0,0.0,0.364831,0.0,0.0,0.0,0.0,0.605693,0.0,0.0,0.0,0.0,0.0,0.0,0.394522,1.49599,0.0,0.0,0.0,0.0385017,0.0,0.0,0.194899,0.0,1.92516,0.0,0.91812,0.633816,0.189573,0.0,0.454673,0.0,2.12027,0.0,2.95705,0.0,1.09205,0.0,0.0,0.0,1.91776,0.806545,2.59111,0.518859,0.819681,0.0,0.0,0.185717,0.0,1.67609,0.954575,0.0,0.0,0.0,0.0,0.0,0.0,0.346025,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.634889,0.976693,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.38498,3.3027,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.135989,0.0,0.0,0.0,1.27672,0.278735,0.0,0.0,0.0,0.0,0.0,1.04568,1.12498,0.0,0.0,0.309003,0.0,0.676417,0.0,0.339069,0.0,0.0,0.0,0.921405,1.62516,0.313113,0.544058,0.14903,0.0,0.0,0.0,0.446102,0.497322,0.316213,0.937838,0.0,0.0,0.0,0.0,0.378832,0.983343,0.84555,0.758229,0.571707,0.0,0.474914,0.0,0.069546,0.574039,0.841495,1.09805,1.3654,0.680642,0.776779,0.0,0.0,0.91623,0.761164,1.80996,0.867927,0.666008,0.92974,0.0775068,0.0,0.0,0.395333,2.28513,1.00684,0.810453,0.0,0.86563,0.360487,0.0,0.0,2.13971,0.770241,0.655675,0.0,0.0,0.0,0.283431,0.0,1.22703,0.83465,0.300266,0.275892,0.371742,0.0,0.0,0.764906,0.657721,0.0,0.0,0.154693,0.0,1.00463,1.46806,0.0,0.0,0.0,0.0,0.933005,0.0,0.542681,1.39394,0.0,0.0,0.524454,0.699309,0.411463,0.0,0.0,0.810048,0.0,0.462887,0.958972,0.0,0.0,0.0,0.0,0.394635,0.0,0.435834,0.0,0.140994,0.0721057,0.0,0.0,0.0,0.0,0.0,0.0,0.971539,2.77169,0.0,0.602401,0.0,0.0,0.0,0.483884,2.51405,1.09329,0.314896,1.11363,0.0,0.0,1.00043,0.0,0.973346,0.0,0.0,1.03544,0.830508,0.607749,1.04898,0.131732,0.257495,0.549663,2.57876,0.588334,0.197826,0.369059,0.84212,0.502513,1.39704,3.09683,1.73015,0.0,0.0,0.0626536,1.14466,3.76175,2.77878,0.647761,0.0,0.0,0.0,0.064981,1.90465,3.41808,0.0970752,0.0,0.0489397,0.0,0.0,0.0,3.88672,1.36897,0.256667,0.0,0.0,0.0,0.0,0.0,1.33696,0.0,0.822399,0.0,0.0,0.258478,0.0,0.0,0.0,0.177872,0.542898,0.00695529,0.115935,0.291098,0.0,0.0,0.60663,0.0963543,0.235644,0.0,0.155717,0.0,0.0114679,0.0,0.0,0.0,0.0,0.0,1.91041,0.231221,0.0,0.0,0.0,0.0,0.0,0.0,1.8309,0.199714,0.0,0.0,0.0,0.570446,0.0,0.0,0.0,0.0,0.0,0.0,0.212868,1.08711,0.0,0.0,0.0,0.665435,0.0,0.0,0.4554,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.47315,0.00822672,0.0,0.0,0.0,0.0,0.0,0.0,1.64584,0.74689,0.0,0.0,0.0,0.0,2.24645,0.0,0.670896,0.151218,0.0,0.0,0.0,0.0132566,2.21195,0.662939,0.758952,0.774986,0.233912,0.192956,0.0,0.699342,1.79039,0.372087,0.0,0.0,0.0,0.0,0.0,0.333497,1.50809,0.523412,0.0,0.0,0.0,0.0,0.0,0.378792,1.51155,0.592428,0.0,0.0,0.0,0.0,0.761351,0.0,1.35299,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.52999,0.0,0.0,0.0,1.60606,0.0,0.0,0.0,0.430976,0.0,0.192244,0.195048,1.01287,1.11199,0.590627,0.0,0.897152,0.0,0.0,0.0,0.330682,0.482386,2.13227,1.84158,2.01878,0.0072947,0.0,0.0,0.0,0.580694,1.14582,0.0,0.0,0.0,0.0,0.0,0.0,0.471932,0.351768,0.0,0.0,0.238902,0.0,0.0,0.0615935,0.0,0.699366,0.0,0.0,0.0,0.0,0.0445606,0.100059,0.0,0.0,0.0,0.0,0.0,0.0600438,0.0474606,0.0,0.0,0.462316,0.0,0.0,0.0,0.189246,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0668448,0.0,0.0,0.0,0.0,1.86483,1.53056,0.0,0.622024,0.284832,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.432719,0.0,0.0,0.0,0.0,0.162582,1.31636,2.15929,1.32799,0.0,0.0,0.0,0.0,0.755417,1.30695,0.380251,0.686545,0.190523,0.0,0.0,0.0,0.370541,0.0442432,0.0,1.81384,1.85696,0.0,0.0,0.442134,0.333907,0.0,0.0,0.31712,1.59907,1.90484,0.0,0.0,0.0,0.0,0.0,0.0,0.206794,3.02312,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.268096,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.528953,1.00114,0.30434,0.0,0.0,0.496645,0.469882,0.0,0.0,0.0,1.52416,0.0,0.195526,0.328009,0.0,0.0,0.0,0.0,0.842644,0.390838,0.184386,0.0,0.0,0.0,0.0,0.671021,0.380011,1.98802,0.497786,0.0,0.0,0.0,0.0,0.0,0.0,1.09613,0.903953,0.0,0.0,0.567454,0.0,0.0,0.0,0.0,0.117787,0.0,0.319587,0.0,0.0,0.0,0.0,0.0,0.990812,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.489345,0.0,0.127618,0.0,0.0,0.0,0.233593,0.055593,0.126105,0.0671247,0.0,0.0,0.0,0.0,0.0,0.26459,0.311991,0.245591,0.0,0.0,0.0,0.174278,0.505018,0.0498403,0.307817,0.0,0.0,0.112105,0.546889,0.0,0.0,0.473316,0.178181,0.0,0.0,0.447637,0.287689,0.0,0.143913,0.0322882,0.00350056,1.7729,0.508216,0.0,0.0,0.634541,0.217582,0.412284,0.0,2.63438,0.0,0.0,0.0933313,0.0,0.0,0.697267,0.0,1.91044,0.167753,0.382487,0.25543,0.257874,0.0,0.0,0.0,0.0,0.587148,0.803546,0.960634,0.0,0.0,0.149105,0.0567676,0.0,0.305464,0.484381,0.0,0.894348,0.0,0.0611869,0.34288,0.0,0.0,0.0769057,0.176249,0.0,0.0,0.196719,0.327177,0.0,0.0,1.00698,0.368367,1.10658,1.61091,0.0,0.21822,0.0,0.469958,0.386309,0.799658,0.0823689,0.0,0.0,0.11666,0.0,0.632634,0.805531,0.369996,0.550876,0.607642,0.0,0.0,0.0,0.142737,0.0963876,0.846876,1.42819,0.640829,0.150833,0.0,0.0,0.875032,0.0897268,0.726392,0.983278,1.42671,2.42407,0.0,0.632376,1.00782,0.876948,0.456728,0.0,0.0,0.715577,0.763421,0.165796,0.463361,1.42565,2.58363,0.0344218,0.0,0.181827,0.352974,0.0,0.686832,1.94523,0.839072,1.18281,0.0,0.0467408,0.375894,0.0,0.0,0.336588,0.0,0.0,0.0,0.0,0.120486,0.0,0.29606,0.148303,0.0,0.791523,0.270257,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.601195,0.0,0.0,0.0,0.0,0.349596,0.0,0.0,0.0,0.103109,0.0,0.0,0.475789,0.608953,0.588312,0.388948,0.0,0.0,0.0,0.753991,0.872843,0.379611,0.0583248,0.0,0.0,0.0,0.807646,0.569583,0.621658,0.0,0.0,0.0,0.0,0.0,0.675389,1.62518,0.0,0.185609,0.940531,1.24809,0.0596206,0.0985181,0.641916,0.947359,0.0,1.42637,0.833612,0.67154,0.0,0.0,0.480338,0.459213,1.06753,0.836595,0.639153,1.031,2.18175,0.0,0.0365831,1.42213,0.425339,0.0,0.0886849,1.29513,0.948854,1.37071,0.762735,1.57083,0.0,0.0,1.711,1.6629,1.46155,0.926492,0.0,2.29242,0.228329,0.0,1.22157,1.78688,0.78242,0.471001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.369135,0.0,0.0,0.0,0.0,0.275048,0.372256,0.420823,0.160078,0.0,0.0,0.0,0.0,0.226786,0.309226,0.423639,0.265677,1.25954,0.406419,0.568785,0.0,0.0,0.964804,0.430798,1.36709,0.833677,0.356695,0.427255,0.0387035,0.0,0.0,0.281644,2.06004,0.510493,0.570369,0.0,0.386908,0.141464,0.0,0.0,1.93393,0.79488,0.239042,0.0,0.0,0.0,0.0,0.0660774,0.841816,0.464318,0.15006,0.0,0.0,0.0,0.0,0.556436,0.0,0.263721,0.204416,0.0,0.438497,0.0,0.0,0.848509,0.0,0.0,0.0,0.0,0.378166,0.0,0.0,1.18219,0.0806748,0.0,0.0,0.0,0.0237719,1.31063,0.339314,1.19487,0.0,0.0,0.0,0.0,0.0,0.0,0.956191,1.10763,0.0,0.0,0.0,0.0,0.0,0.630922,2.31696,0.679529,0.0,0.0,0.0,0.0,0.0,0.0,3.0052,2.42187,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.03137,0.0,0.11117,0.0,0.0,0.152857,0.0,0.0,0.438231,0.0,0.857186,1.20607,1.37037,0.412456,0.0,1.05695,1.1725,0.128264,1.79321,2.12909,1.35975,0.438601,0.0,1.82074,1.72758,0.182807,0.602843,0.0,0.0,0.0,0.0,0.909633,1.73396,0.593386,0.0,0.219959,0.317044,0.0,0.206743,0.0,1.67549,0.0,0.0,0.0,0.0521286,0.0,0.0,0.0,1.49735,0.0,0.324238,0.0,1.10249,0.0,0.205561,0.0,0.0,0.0,0.0,0.0,0.533768,0.35955,0.0,0.0,0.0,0.0,0.0599714,0.615134,0.615925,0.481685,0.822199,0.500898,1.16356,0.0,0.0,0.0,0.0,1.17843,0.103381,1.18782,0.0,0.0,0.0,0.631129,1.97183,0.759947,0.0,1.63086,0.0,0.0,2.70253,1.74042,0.172807,0.00409326,0.0,0.0762443,0.0,1.58742,2.60939,0.115389,0.0648634,0.0,0.0,0.0,0.0,2.0035,0.353461,0.10496,0.194036,0.0,0.0,0.0,0.0,0.0,0.297805,0.0,2.64653,0.0,1.23877,0.0,0.0,0.0,0.169143,2.34808,0.419412,0.0,0.409767,0.0,0.0,0.0,0.282705,1.27803,0.0687246,0.0,0.134419,0.677217,1.28184,0.0,1.38181,1.72411,1.8383,0.0,0.0,0.4532,1.02912,0.152686,1.95838,1.69392,0.754414,0.0,0.0,0.46667,1.32661,0.0254744,0.0,0.388074,0.580704,0.698675,0.0,0.829627,1.30728,0.0,0.0,1.2511,0.93468,0.598455,0.0,0.0,1.2374,0.0,0.274115,0.446321,0.604911,0.795081,0.615409,0.0,0.888758,0.0,0.375484,0.0,0.0,0.837868,0.52996,0.0,0.0,0.0,0.0,0.0,0.175929,0.454577,0.0,0.1911,0.0,0.0,0.397681,0.32386,1.20274,1.07782,0.0,0.2635,0.0,1.33057,0.491385,0.0,0.0,0.275172,0.219227,0.650845,1.35379,1.3222,0.0,0.0,0.0,0.0,0.880018,0.63206,1.09638,1.19715,0.663006,0.0,0.0,0.0,0.923722,0.0,1.09659,0.393327,1.03734,0.0,0.0658347,0.0,0.0,0.794975,0.904895,0.853687,0.243112,0.397111,0.313996,0.0754557,0.0,0.0,0.364119,1.27006,0.0262004,1.73156,0.585787,0.305033,0.0,0.0,0.0,0.678363,0.191614,1.53553,0.413996,0.476185,0.547662,1.20803,1.4871,0.0,0.170995,0.689346,0.519208,0.399542,1.23518,0.5084,2.1031,0.699747,0.0,0.0,0.0,0.0,0.135941,0.270758,0.0,0.436416,0.0,0.0,0.0,0.0,0.749222,0.0,0.0,0.0,0.0,0.0,0.0,0.0183226,2.31644,0.440846,0.0,0.0,0.0,0.0,0.0,0.0,0.0267736,2.01551,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.479,0.0999095,0.359901,0.0,0.0,0.00408712,0.0,0.0,1.56113,2.43366,0.704628,0.0,0.0533517,0.0,0.0,0.0889012,0.0,1.14794,0.127724,0.0,0.0,0.0,0.0,0.0821292,0.0,0.322749,0.357619,0.198481,0.37144,0.731281,0.0,0.0,0.709205,0.0,0.377677,0.0,0.0,0.0,0.0,0.0,0.33062,0.0,0.0,0.0,0.0,0.234784,0.430148,0.0,1.62973,0.0,0.0,0.0,0.0918751,0.0,1.25778,0.0,0.420295,0.0,0.0,0.0,0.0,0.0,0.765266,0.0,0.0,0.492427,0.0,0.0,0.0,0.0,0.0,0.0968297,0.0,0.820264,0.0,0.0,0.0,0.89482,0.87781,0.817386,0.0,0.0,0.0,0.0,0.0,0.153692,0.452499,0.636759,1.14697,0.0,1.83968,0.202893,0.0554429,0.387769,0.0,1.27937,1.69342,0.114803,1.52515,0.59907,0.0,0.371332,0.0,0.52635,1.5612,0.059289,0.241491,0.0,0.0,0.0796511,0.0,0.0,1.52604,0.0590937,0.0,0.0,0.559116,0.785958,0.0,0.0,1.69145,0.0933163,0.204759,0.514798,0.61769,0.0701729,1.10675,0.0,0.0,1.06784,1.33991,0.0,0.0,0.0,0.42707,0.503573,0.0,0.0,0.670961,0.0,1.4024,0.0,0.0,0.0,0.0,0.0,0.962079,0.0,1.1311,0.0,0.0,0.0,0.0,0.0,0.0,0.921526,0.906567,0.670576,1.2226,0.38453,0.34655,0.975119,0.0,0.409371,0.951431,1.09325,1.18589,0.0,0.0,0.760578,0.0,0.756743,0.635672,0.0940952,0.0808923,0.0,0.0,0.767397,0.113031,0.00293017,0.221256,0.0757609,0.126439,1.08849,0.0,0.60052,0.0,0.00478263,0.11381,0.15906,0.0,0.267037,0.0,0.331809,0.0,0.0698263,0.0543275,0.307899,0.0,0.0327412,0.0,0.0,0.0,0.0,0.0,0.105932,0.691574,0.334606,0.574586,0.0,0.0,0.627389,0.212618,0.592801,0.641913,1.05172,1.45989,0.581127,0.0593375,0.0,0.0,0.0,0.0,0.476081,0.0,0.0,0.823936,0.0,0.0,0.0,0.0,0.960093,0.0,0.245482,0.829666,0.0,0.0,0.0,0.0,1.59576,0.654682,0.232491,0.0,0.0,0.0,0.210807,0.507086,0.0,1.89822,0.380005,0.0,0.0,0.0,0.0,0.668549,0.0,1.59211,1.02978,0.210456,0.0,0.579386,0.0,0.0145059,0.0,0.566487,1.33053,1.21942,0.13497,0.112133,0.0,0.0,0.467411,0.0,0.900559,0.627929,0.239716,0.356065,0.0,0.0,0.0,0.0,0.11928,0.0,1.31571,1.20622,0.727616,0.0,0.0,0.0,0.868163,0.0753211,0.828723,0.61893,0.0,0.0,0.0,0.0,1.10761,0.0,0.0,0.174353,0.512423,0.964466,0.0,0.474061,1.08443,0.0,0.0,0.956998,0.638231,0.134535,0.0,0.0,0.946838,0.0,0.0613044,0.234921,0.342171,1.09688,1.26384,0.0,0.23803,0.0,0.330917,0.0,0.0,0.600456,0.106206,0.878113,0.0363169,0.0,0.0,0.0,0.260346,0.61787,0.0,0.365467,0.0,0.0,0.0,0.190987,1.31754,1.04369,0.0,0.23753,0.0,0.0,0.0,0.0,0.0,0.0,0.1596,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.604422,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.202996,0.0,0.0,0.0,0.44572,0.0,0.0,0.0,0.0,0.271033,0.0,0.492986,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.21595,0.0,1.02876,0.451589,0.0,0.0,0.0,0.0,1.52587,0.598599,0.901327,0.0,0.0,0.278006,0.0,1.02077,1.21872,0.013687,0.0,0.0,0.0,0.755327,0.348266,0.827607,0.209397,0.0635253,0.218324,0.498259,1.18562,1.0566,0.266579,0.0,0.0,0.350892,0.830991,1.34277,1.20945,0.349505,0.0,0.0,0.0103536,0.361914,0.739187,0.315481,0.460837,2.31026,0.01676,0.0,0.254955,0.412521,0.0,0.0,1.55604,0.820949,0.615958,0.0,0.220733,0.0,0.0,0.0,0.0,1.44878,4.073,0.0,0.0,0.0,0.0,0.0,0.0,0.27648,4.63813,2.45928,0.0,0.0,0.0,0.0,0.0,0.0,0.10872,0.461444,0.0,0.0788399,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.29525,0.301444,0.0,0.328148,0.0,0.0,1.11521,1.24517,1.0101,0.0,0.0,0.0521027,0.259551,2.8857,0.675328,0.0,0.0,0.763001,0.0,0.0755973,2.60212,1.21681,0.0,0.0,0.0500128,1.01414,0.0,0.0882898,0.985881,0.0,0.0,0.0,0.0,0.0,1.61944,0.0,0.0,0.0,0.0,2.41404,0.0,0.0539209,1.12438,1.89864,0.0,0.0,0.578968,1.36058,0.597613,0.45964,0.0,0.657914,0.0,0.0,0.0374687,0.0,0.0978104,1.62391,0.130971,1.3785,0.885469,1.56421,1.75278,1.76503,0.100433,0.0,0.828363,1.36913,0.667373,1.50033,1.37664,0.975089,0.0,0.0,0.477156,1.27557,1.00322,0.924455,0.680938,0.0,0.190288,0.0,0.949042,1.23412,0.55296,0.0,0.487067,0.130915,0.0,0.624962,0.0,1.09289,0.378908,0.173529,0.15106,0.00739,0.261807,0.58442,0.0,0.917083,0.0,0.334967,0.0,0.283461,0.310288,0.0,0.0,0.156471,0.0,0.0,0.0,0.676767,0.957265,0.458896,0.00670796,0.0,0.346182,0.0,0.0,0.907569,0.919031,0.687937,0.94205,0.0,0.0,0.0,0.0,0.0,0.113613,0.261331,0.0,0.0,0.0,0.122713,0.000806917,1.17298,1.2273,0.12292,0.131726,0.0,0.0,0.423984,0.00841113,0.0,0.0,0.0,0.711064,0.0,0.0,0.0,0.0,0.0,2.22146,2.09239,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.34187,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.033636,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.615348,0.0,0.0,0.0,0.0,0.0,0.0,0.641062,0.0,1.17372,0.495191,0.442738,0.614823,0.0,0.926129,1.44889,0.473809,1.03521,0.0,0.0,0.0,0.0,1.54896,1.42159,0.411781,0.16762,0.0,0.0,0.0,0.0,0.0,1.55387,0.393557,0.0,0.0,0.0,0.0,0.0,0.0,1.58697,0.571905,0.0,0.0,0.11285,0.0,0.0,0.0,0.0,1.10844,0.0,0.0264758,1.39826,0.0,0.0,0.0,0.0,0.0,0.0,0.58123,1.25104,0.0,0.0,0.411561,0.0,0.513784,0.0,0.459934,0.337147,0.0,0.0,0.728633,0.965154,1.40545,0.162564,0.0,0.0,0.186592,0.0,0.363265,0.0,0.301644,0.377937,0.0686808,0.0,0.429569,0.0229015,1.50169,0.0,0.275216,1.30152,0.0,0.0,1.13964,0.792563,2.35811,0.0,0.295911,1.23007,0.0,0.453664,0.502166,1.24174,1.05941,0.946884,0.161656,0.0,0.0,0.616832,0.78498,0.76343,1.16599,2.29191,0.431154,0.0,0.584949,2.15721,0.6331,0.522089,1.0079,2.30621,1.95331,0.153324,1.45694,2.06909,0.348056,0.368991,1.21501,0.711645,1.28046,0.521661,1.89765,1.10906,0.212319,0.41663,0.812447,0.879762,0.907212,0.0,0.400943,0.468782,0.516605,1.31639,1.07805,0.577831,0.265595,0.0,0.191645,1.40947,1.25448,1.02759,0.162584,0.474091,0.127062,0.0,1.5603,1.38745,0.737667,0.151507,0.0,0.0,0.139331,0.714281,1.45387,0.796915,0.623296,0.0,0.0,0.0,0.0,0.678722,0.7811,0.615184,1.14835,0.0482833,0.0,0.0,0.0,0.0,1.28849,0.86627,0.0,0.794121,0.976327,0.0,0.0,0.0,0.945367,2.01491,0.0,0.0,0.0,0.291776,0.0,0.0,1.38724,0.946639,0.466857,0.0785764,0.0,0.0485555,0.914422,0.828808,1.64758,1.84144,1.65575,0.464215,0.0,1.11952,1.69265,0.682919,1.35612,1.06472,0.0,0.0,0.0,1.13254,1.58718,0.5078,0.0,0.0950873,0.0374254,0.623689,0.0,0.772888,1.57999,0.0,0.0,1.12806,0.76438,0.0,0.0,0.0,1.46127,0.0,0.555015,0.546193,0.573393,0.841144,0.436939,0.0,1.20008,0.0,0.64314,0.0,0.085535,0.810178,0.943781,0.0,0.0,0.0,0.0,0.0,0.460152,0.498044,0.0,0.240839,0.0,0.0,0.0816919,1.05791,1.48168,1.19605,0.0,0.0,0.17918,1.12625,0.0,0.0,0.7285,0.0,0.838999,0.861548,0.129623,0.83792,0.175348,0.0,1.54106,0.0,0.163841,0.347958,0.0,1.22459,1.89909,1.53905,1.30996,0.0,0.0,0.855998,0.0,1.66283,0.9224,0.0570119,0.0937239,0.0,0.0,0.689249,0.101791,2.00403,0.290622,0.0,0.06935,0.606224,0.0,0.143588,1.34034,2.55846,0.0,0.0,0.0,0.0,0.291031,0.0,0.633558,1.6714,0.817377,0.625381,0.0,0.0,0.0,0.0,0.191478,2.23494,0.0784995,0.131453,0.0,0.0,0.0,0.0,0.0,0.190803,1.28915,1.41198,1.01767,0.401728,0.0,0.596131,1.71468,0.301218,0.560684,0.414054,0.0,0.0,0.0,0.579402,1.75805,0.525132,0.816719,0.143245,0.0,0.014217,0.0,0.173443,1.74813,0.476386,0.171834,0.49605,0.0456828,0.137179,0.201566,0.0,1.62037,0.00669221,0.0,0.0,0.282296,0.233941,0.644033,0.0,0.694103,0.0,0.0,0.0,1.49962,0.0742687,0.440048,0.179104,0.458199,0.0,0.0,0.0,2.25741,1.90252,0.720989,0.0,0.0,0.0,0.0,0.0,1.44913,1.87545,1.76967,1.39909,0.0,1.53672,1.52084,1.27963,1.08633,0.903548,1.60705,1.22292,1.43142,0.545284,0.0,0.0,0.0,0.239591,0.562282,0.234012,0.421088,0.658034,0.0,0.108445,0.266689,0.452471,1.79748,0.311373,0.441496,0.502177,0.188584,0.198001,0.195473,0.295144,0.0,0.469018,0.345084,0.0,0.0,0.733671,0.6339,0.0,0.178559,0.401977,0.341566,0.0,0.107567,1.30035,0.700756,0.377124,0.89293,0.0427921,1.49304,0.0,0.0596879,0.843327,1.52879,1.39502,1.37738,0.206584,0.942413,0.0,1.22737,1.1419,0.887947,1.51065,1.25406,1.16337,0.613377,0.605455,0.0,0.24666,0.868317,0.0,0.360846,0.936186,0.0022287,0.536991,0.250131,1.20433,2.28587,0.0,0.0,1.05447,0.0,0.0,1.61513,1.13274,0.0,0.0,0.0,0.650125,0.0,0.0,0.399529,0.0,0.0,0.0,0.0,0.740964,0.032971,0.490365,0.0,0.0,0.0,0.111532,0.0,0.0,0.680296,0.564668,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.691355,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.350713,0.0,0.0,0.0,0.0,0.0161727,0.191207,0.0,0.337782,1.12287,0.649489,0.400047,0.0,0.0,0.183137,0.399125,0.843159,0.0,0.0,1.05744,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.48364,0.0,0.0,0.0,0.0,0.0,0.0,0.270507,1.41753,0.00219141,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.30818,0.0199869,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.114416,0.251498,0.0,0.380314,0.286347,0.0,0.0,0.0,0.0,0.0105545,0.186821,1.12282,2.08228,2.74071,0.645817,0.0,0.0,1.68138,0.69736,2.66767,2.69961,0.155789,0.105847,0.0,0.248712,1.95955,1.09794,0.0,0.0,0.490212,0.639648,0.0,0.439032,1.9401,0.150369,0.0,1.18878,1.01829,0.927629,0.361353,0.154466,1.99311,0.0,0.401436,0.347202,0.56683,0.930807,1.229,0.0,1.52818,0.0,0.561121,0.0,0.0,0.966648,0.454618,0.599211,0.0838991,0.0,0.0,0.0,0.0268369,0.145766,0.0,0.933924,0.0,0.0,0.0387872,0.192547,1.25894,1.02336,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.131825,0.0,0.0744442,0.0,0.0,0.0,0.0,0.0,0.538258,0.108265,0.0,0.0,0.0,0.0,0.0260937,0.0,0.289256,0.118882,0.0,0.0,0.569441,0.469531,0.0,0.0,0.0,0.147205,0.0,0.0531026,0.33969,0.605172,0.264042,0.0,0.0,0.0305189,0.0,0.510273,0.0,0.0,0.19607,0.125283,0.0,0.0,0.0,0.100206,0.112939,0.0,0.0,0.0,0.0,0.0,0.0,0.696627,0.642053,0.274481,0.163283,0.0,0.0,0.299046,0.987159,0.341216,0.0,0.0,0.0,0.0,0.196597,0.199021,1.55231,0.0,0.0,0.0,0.0,0.0,0.667122,0.843999,1.08041,0.0,0.0,0.00574162,0.28691,1.06219,1.01786,0.815518,0.0,0.0,1.30409,1.18415,0.0,0.0,0.35683,0.718967,0.0,0.783926,0.842383,0.846945,1.24251,0.905805,0.200497,0.445207,1.32511,0.787197,0.0,0.0,1.12959,0.580888,0.622787,0.483673,1.66148,0.0,0.0,0.512424,0.125854,0.256264,0.520314,0.0,1.02362,0.0979093,0.679154,0.840328,0.863938,0.0792286,0.0,0.0,0.0,0.0,0.0,0.0,0.742585,0.300129,0.0,0.0,0.0,0.0,0.392517,0.0,0.0,0.832888,0.0778058,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.33371,0.0,0.0,0.0,0.0,0.995592,0.705507,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.572888,0.62149,0.0,0.0202326,1.41213,0.57087,0.0,0.0,0.0,0.929488,0.0,1.16965,2.06719,0.0,0.0,0.0,0.0,0.866699,0.0,0.424763,0.0,0.0,0.0,0.658806,0.333852,1.38387,1.05636,0.784783,1.64189,2.47508,1.58976,0.0,0.753668,1.2476,1.20182,2.50585,2.92566,1.78411,0.369189,0.359947,0.741087,1.21331,1.62756,0.552454,0.0613478,0.0,0.0,0.0,1.32823,1.19699,1.76264,0.0,0.0,0.0,1.81481,1.62417,0.727542,1.42801,0.0,0.0,0.0,0.0,0.0,0.0,0.573907,1.5855,0.0,0.0,0.0,0.865401,0.0,0.00663826,0.0,0.950973,0.0,0.0,0.0,1.10933,0.835694,0.544583,0.0,0.0,0.0,0.0,0.0,0.26545,0.610689,1.15868,0.286849,0.0,1.30336,0.794415,0.46181,0.0,0.0,0.0,1.07866,1.66779,1.52057,0.405416,0.0,0.0,0.0,0.0,1.43493,1.87474,1.83861,0.0,0.0,0.564343,0.371479,0.335707,1.05658,1.86748,1.94895,0.0,0.737799,0.986488,0.0,0.0,0.79844,1.65232,0.0,0.0,0.454446,0.662035,0.917331,0.950358,0.0,1.46954,0.658963,0.724276,0.0,1.20658,0.644188,0.865547,0.839145,1.62406,0.646788,0.0,0.0,1.33134,0.195276,0.0163854,0.226701,0.234888,1.33465,0.0,0.999181,1.28773,1.1352,0.0,0.0,0.518403


	};
	vector<Mat> W, X;
	
	for (int i = 0; i < 64; i++) {
		double mm[9];
		for (int j = 0; j < 9; j++) {
			mm[j] = w1[i * 9 + j];
		}
		W.push_back(Mat(1, 3, 3, vector<double>(mm, mm + 9)));		
	}
	X.push_back(Mat(64, 8, 8, vector<double>(input, input + 64 * 8*8)));
	
	DW_Convolution cv(W, 1, 1);
	auto e = cv.forward(X);
	int cnt = 0;
	for (int i = 0; i < e.size(); i++) {
		for (int d = 0; d < e[i].dim; d++) {
			for (int r = 0; r < e[i].row; r++) {
				for (int c = 0; c < e[i].col; c++) {
					printf("%lf ", e[i].mat[d*(e[i].row*e[i].col) + r*e[i].col + c]);
					cnt++;
				}
				printf("\n");
			}
			printf("\n");
		}
	}
	printf("cnt : %d\n", cnt);



	//vector< vector< unsigned char > > test_images
	//				 3072x
	//
	//double data[4] = { 1.0, 2.0, 3.0, 4.0 };
	//Mat a = Mat(new int[2]{ 2,2 }, data);
	//a.print();
	/*ifstream f("params2_back.csv");
	string line, key, shape;*/
	//while (getline(f, line, ',')) {
	//	size_t prev=0,pos;
	//	
	//	if ((pos = line.find_first_of("\n", prev)) != string::npos)
	//	{	
	//		cout << "??" << endl;
	//		cout << line.substr(prev, pos - prev) << endl;;
	//		prev = pos + 1;
	//		cout << line.substr(prev, string::npos) << endl;
	//	}
	//	else {
	//		//cout << line.length() << endl;
	//		//cout << line[0] << endl;
	//		cout << line  << endl;
	//	}
	//	
	//}
	return 0;
}