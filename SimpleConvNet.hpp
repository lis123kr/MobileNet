#ifndef SIMPLECONV_
#define SIMPLECONV_
#include"Layers.hpp"
#include<vector>
struct input_dim {
	int d1, d2, d3;
};

struct conv_param {
	int f1, f2, f3;
	int filtersize, pad, stride;
};

class SimpleConvNet {
private:
	std::vector< Layer* > layers;
public:
	SimpleConvNet() {}
	~SimpleConvNet() {}
	SimpleConvNet(input_dim id, conv_param cp, int hidden_size=512, int output_size=10, bool pretrained=true) {

		layers.push_back(new DW_Convolution());

	}

	std::vector< Mat2D > predict(std::vector<Mat2D> x) {
		for (int i = 0; i < layers.size(); i++) {
			x = layers[i]->forward(x);
		}
		// 수정 필요
		return x;
	}

	int accuracy(std::vector< std::vector< unsigned char > > x, std::vector< int > t, int batch_size=100) {

		return 0.1;
	}

};

#endif
