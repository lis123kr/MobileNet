#ifndef LAYERS_
#define LAYERS_
#include<vector>

// 3D Matrix
struct Mat {
	int dim, row, col;
	std::vector< double > mat;
	Mat(int dim, int row, int col, std::vector< double > v) : dim(dim), row(row), col(col), mat(v) {};
};

//typedef std::vector< std::vector< std::vector< double> > > Mat3D;
//typedef std::vector< std::vector< double> > Mat2D;
//typedef std::vector< double > Mat1D;

// Mat 일차원으로 계산 + dimension 변수
class Layer {
public:
	virtual std::vector<Mat> forward(const std::vector<Mat> &x) { return std::vector<Mat>(); }
};


// padding
class Convolution : public Layer {
private:
	std::vector<Mat> W;
	int stride, pad;
public:
	Convolution() {};
	~Convolution() {};
	Convolution(std::vector<Mat> W, int stride=1, int pad=0) : W(W), stride(stride), pad(pad) {};

	// Each image x conv with each w
	virtual std::vector<Mat> forward(const std::vector<Mat> &x) {
		std::vector< Mat > out;
		int n, nw;
		for (n = 0; n < x.size(); n++) {
			std::vector<double>rev;
			for (nw = 0; nw < W.size(); nw++) {
				auto e = Convolution_(x[n], W[nw]);
				rev.insert(rev.end(), e.begin(), e.end());
			}
			int out_r = (x[n].row + 2 * pad - W[0].row) / stride + 1;
			int out_c = (x[n].col + 2 * pad - W[0].col) / stride + 1;
			out.push_back(Mat(nw, out_r, out_c, rev));
		}
		return out;
	}

	// Convolution x and W (both are 3-D Mat)
	std::vector<double> Convolution_(const Mat& x,const Mat& w) {
		std::vector<double> ret;
		int ndim = x.dim - w.dim + 1;
		for (int d = 0; d < x.dim - w.dim + 1; d++) {
			for (int r = -pad; r < x.row - w.row + 1 + pad; r++) {
				for (int c = -pad; c < x.col - w.col +1 +pad; c++) {
					ret.push_back(Convolution_(x, w, d, r, c));
				}
			}
		}
		return ret;
	}

	double Convolution_(const Mat& x, const Mat& w, int d, int r,int c) {
		double ret = 0, xx=0;
		int ds = w.col * w.row, rs = w.col;
		int dxs = x.col * x.row, rxs = x.col;
		for (int dd = 0; dd < w.dim; dd++) {
			for (int rr = 0; rr < w.row; rr++) {
				for (int cc = 0; cc < w.col; cc++) {
					if ((pad > 0) && (r + rr < 0 || c + cc < 0 || r + rr >= x.row || c + cc >= x.col))
						xx = 0;					
					else
						xx = x.mat[(d + dd)*(dxs)+(r + rr)*rxs + (c + cc)];
					ret += xx * w.mat[dd*(ds)+rr*(rs)+cc];					
				}
			}
		}
		return ret;
	}
};

// Depthwise Conv
class DW_Convolution : public Layer {
private:
	std::vector<Mat> W;
	int stride, pad;
public:
	DW_Convolution() {};
	~DW_Convolution() {};
	DW_Convolution(std::vector<Mat> W, int stride=1, int pad=0) : W(W), stride(stride), pad(pad) {};

	virtual std::vector<Mat> forward(std::vector<Mat> &x) {
		std::vector<Mat> out;
		int n, d;
		for (n = 0; n < x.size(); n++) {
			// Each dimension Conv with each filter
			std::vector<double>rev;			
			for (d = 0; d < x[n].dim; d++) {
				auto e = Convolution_(x[n], W[d], d);
				rev.insert(rev.end(), e.begin(), e.end());
			}
			int out_r = (x[n].row + 2 * pad - W[0].row) / stride + 1;
			int out_c = (x[n].col + 2 * pad - W[0].col) / stride + 1;
			out.push_back(Mat(d, out_r, out_c, rev));
		}
		return out;
	}

	std::vector<double> Convolution_(const Mat& x, const Mat& w, int d) {
		std::vector<double> out;
		int dd = d * x.col * x.row;
		for (int r = -pad; r < x.row - w.row + 1 + pad; r++) {
			for (int c = -pad; c < x.col - w.col + 1 + pad; c++) {
				out.push_back(Convolution_(x, w, dd, r, c));
			}
		}
		return out;
	}

	double Convolution_(const Mat& x, const Mat& w, int dd, int r, int c) {
		double ret = 0, xx=0;
		for (int rr = 0; rr < w.row; rr++) {
			for (int cc = 0; cc < w.col; cc++) {
				if ((pad > 0) && (r + rr < 0 || c + cc < 0 || r + rr >= x.row || c + cc >= x.col))
					xx = 0;
				else
					xx = x.mat[dd + (r + rr)*x.col + (c+cc)];					
				ret += xx * w.mat[rr*w.col + cc];
			}
		}
		return ret;
	}
};

class Relu : public Layer {
public:

};

//class Affine {
//private:
//	Mat W;
//	int col, row;
//public:
//	Affine() {}
//	~Affine() {}
//	Affine(Mat W) {
//		this->W = W;
//		this->row = W.size();
//		this->col = W[0].size();
//	}
//	// x.shape = [512]
//	Mat1D forward(Mat1D& x) {
//		Mat1D out = Mat1D(col);
//		for (int c = 0; c < col; c++) {
//			int dot_ = 0;
//			for (int r = 0; r < row; r++) {
//				dot_ += x[r] * W[r][c];
//			}
//			out[c] = dot_;
//		}
//		return out;
//	}
//};
//
//class Pooling {
//private:
//	int stride, pool_w, pool_h, pad;
//public:
//	Pooling() { pad = 0; };
//	~Pooling() {};
//	Pooling(int pool_h, int pool_w, int stride=1, int pad=0) :pool_h(pool_h), pool_w(pool_w), stride(stride), pad(pad) {};
//	Mat forward(Mat x) {
//		int row = x.size(), col = x[0].size();
//
//		int o_h = ((row + 2 * pad - pool_h) / stride) + 1;
//		int o_w = ((col + 2 * pad - pool_w) / stride) + 1;
//		Mat out(o_h);
//		for (int i = 0; i < o_h; i++)
//			out[i] = Mat1D(o_w);
//
//		for (int r = -pad, rr=0; r + pool_h <= row+pad; r+=stride, rr++) {
//			for (int c = -pad, cc=0; c + pool_w <= col+pad; c+=stride, cc++) {
//				int max_ = 0, cmp_;
//				for (int i = 0; i < pool_h; i++) {
//					for (int j = 0; j < pool_w; j++) {
//						if (pad > 0 && (r + i < 0 || c + j < 0 || r+i >=row || c+j >= col))
//							cmp_ = 0;
//						else
//							cmp_ = x[r + i][c + j];
//						if (max_ < cmp_)
//							max_ = cmp_;
//					}
//				}
//				out[rr][cc] = max_;
//			}
//		}
//		return out;
//	}
//};
//
//class LightNormalization {
//private:
//
//public:
//
//};



#endif
