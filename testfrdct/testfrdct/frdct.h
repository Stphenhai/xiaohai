
#ifndef _FRDCT_H
#define _FRDCT_H
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat dctMatrix(int N);
Mat frDct(Mat& I, MatND a, MatND q);
Mat getFrDctCa(int N, MatND a, MatND q);
double getMSE(const Mat& I1, const Mat& I2);
double getPSNR(const Mat& I1, const Mat& I2);

//logitic-- 1维混沌系统
Mat chaos(double seed,  int ind, double u, int  sizeMatrices);

//三维超混沌系统
Mat threeDchaoticmap(double x0,double y0,double z0,double a,double b,double l);

//行列搬移置乱
Mat rotationscrambling(Mat &I,MatND x, MatND y, MatND z);
Mat inverserotationscrambling(Mat &I, MatND x, MatND y, MatND z);   //置乱的逆过程

Mat Arnold(Mat& I1, Mat &C);
Mat iArnold(Mat& I1, Mat &C);

//量化函数(正过程和逆过程)
Mat lianghua(Mat& I1, double Mx, double Mn);
Mat ilianghua(Mat& I1, double Mx, double Mn);

Mat imageXor(Mat& I1, Mat &C);
Mat inverseimageXor(Mat& I1, Mat &C);

int mod(int x, int y);
double Entropy(Mat img);

bool isrp(int a, int b);      //判断两个数
//bool isPrime(unsigned long n);   //判断函数是否为素数
//MatND factor(int N);     //该函数的功能: 因数分解
//MatND sushu(int N);     //此函数的功能：求与N互素的数

//Mat logistic2(double seed1, double seed2, int ind, double u1, double u2, int sizeMatrices);
double pi = atan(1.0)*4.0;


int mod(int x, int y) {
	int out;
	if (x > 0)
		out = fmod(x, y);
	else
		out = fmod(-x, y);
	return out;
}
bool isrp(int a, int b)
{
	if (a == 1 || b == 1)     // 两个正整数中，只有其中一个数值为1，两个正整数为互质数
		return true;
	while (1)
	{          // 求出两个正整数的最大公约数
		int t = a%b;
		if (t == 0)
		{
			break;
		}
		else
		{
			a = b;
			b = t;
		}
	}
	if (b>1)	return false;// 如果最大公约数大于1，表示两个正整数不互质
	else return true;	// 如果最大公约数等于1,表示两个正整数互质
}


//Function returning fractional DCT of an image with fraction a and GS q
Mat frDct(Mat& I, MatND a, MatND q)
{
	int i, j, M, N;
	clock_t duration;
	M = I.rows;
	N = I.cols;
	if (M != N)
	{
		cout << endl << " Height and width of the image must agree, exit";
		exit(0);
	}
	Mat J = Mat(M, N, CV_64FC1);
	duration = clock();
	//对图像行变换  不同的a  q
	MatND a1 = Mat::zeros(N / 2, 1, CV_64FC1);
	MatND q1 = Mat::zeros(N / 2, 1, CV_64FC1);
	MatND a2 = Mat::zeros(N / 2, 1, CV_64FC1);
	MatND q2 = Mat::zeros(N / 2, 1, CV_64FC1);

	for (int i = 0; i < N / 2; i++)
	{
		a1.at<double>(i) = a.at<double>(i);
		a2.at<double>(i) = a.at<double>(i + (N / 2));
		q1.at<double>(i) = q.at<double>(i);
		q2.at<double>(i) = q.at<double>(i + (N / 2));
	}
	
	Mat Ca1 = getFrDctCa(N, a1, q1);	
   //对图像列变换  不同的a  q

	Mat Ca2 = getFrDctCa(N, a2, q2);

	duration = (clock() - duration) / CLOCKS_PER_SEC;
	cout << endl << duration << " seconds for getting an FrDCT matrix of size " << M << " x " << N<<endl;

	//行行变换
	MatND temp = Mat(N, 1, CV_64FC1);
	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
			temp.at<double>(j) = I.at<double>(i, j);
		temp = Ca1*temp;
		for (j = 0; j < N; j++)
			J.at<double>(i, j) = temp.at<double>(j);
	}


	//列列变换
	 temp = Mat(M, 1, CV_64FC1);
	for (j = 0; j < N; j++)
	{
		for (i = 0; i < M; i++)
			temp.at<double>(i) = J.at<double>(i, j);
		temp = Ca2*temp;
		for (i = 0; i < M; i++)
			J.at<double>(i, j) = temp.at<double>(i);
	}

	return J;
}



/*
//Function returning fractional DCT matrix with fraction a and GS q
Mat getFrDctCa(int N, MatND a, MatND q)
//Reference: Cariolaro G, Erseghe T, Kraniauskas P(2002) The Fractional discrete
// cosine transform.IEEE Trans Signal Process 50(4) : 902 - 911
{
	int n, m, i, j, k,l;
	double phi, cos_wn[512][512], sin_wn[512][512];
	double r1, i1, r2, i2, A, B;
	double realEigenValues[1024], imagEigenValues[1024], temp[1024];
	Mat realEigenVectors = Mat(N, N, CV_64FC1), imagEigenVectors = Mat(N, N, CV_64FC1);
	char b[100]; _itoa(N, b, 10);

	if (N > 1024)
	{
		cout << endl << "N must not be larger than 1024";
		exit(0);
	}

	//读取特征值
	char fileName[100];
	strcpy(fileName, "Eigendata/EigenValues");
	strcat(fileName, b);
	strcat(fileName, ".dat");
	FILE* fp;
	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		cout << endl << "File " << fileName << " read error";
		exit(0);
	}
	fread(realEigenValues, sizeof(double), N, fp);
	fread(imagEigenValues, sizeof(double), N, fp);
	fclose(fp);

	//计算特征相角,只取(0,pi)之间的N/2个相角,0,2,4,...(2n)...,
	for (l = 0; l < N; l++)
	{
		for (n = 0, k = 0; n < N / 2; n++, k += 2)
		{
			phi = atan2(imagEigenValues[k], realEigenValues[k]);
			cos_wn[l][n] = 2.0*cos((phi + 2.0*pi*q.at<double>(l,n))*a.at<double>(n));
			sin_wn[l][n] = 2.0*sin((phi + 2.0*pi*q.at<double>(l,n))*a.at<double>(n));
		}
	}
	

	//读取特征向量
	strcpy(fileName, "Eigendata/EigenVectors");
	strcat(fileName, b);
	strcat(fileName, ".dat");
	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		cout << endl << "File " << fileName << " read error";
		exit(0);
	}

	//特征向量的文件格式是，特征向量1的实部,特征向量1的虚部,特征向量2的实部，特征向量2的虚部,......
	for (n = 0; n < N; n++)
	{
		fread(temp, sizeof(double), N, fp);
		for (m = 0; m < N; m++)
			realEigenVectors.at<double>(m, n) = temp[m];
		fread(temp, sizeof(double), N, fp);
		for (m = 0; m < N; m++)
			imagEigenVectors.at<double>(m, n) = temp[m];
	}
	fclose(fp);

	//计算FrDCT矩阵，分数阶a，Gs q,大小NxN
	Mat Ca = Mat::zeros(N, N, CV_64FC1);
	Mat tempCa = Mat(N, N, CV_64FC1);
	cout << endl;
	for (l = 0; l < N; l++) {
		for (n = 0, k = 0; n < N / 2; n++, k += 2)
		{
			cout << "\rComputing Ca: " << n + 1 << " / " << N / 2;
			for (i = 0; i < N; i++)
			{
				r1 = realEigenVectors.at<double>(i, k);
				i1 = imagEigenVectors.at<double>(i, k);
				for (j = i; j < N; j++)
				{
					r2 = realEigenVectors.at<double>(j, k);
					i2 = imagEigenVectors.at<double>(j, k);
					A = (r1*r2 + i1*i2)*cos_wn[l][n];
					B = (r1*i2 - i1*r2)*sin_wn[l][n];
					tempCa.at<double>(i, j) = A + B;
					if (i != j)tempCa.at<double>(j, i) = A - B;
				}
			}
			add(Ca, tempCa, Ca);
		}
	}
	return Ca;
}    */

//Function returning fractional DCT matrix with fraction a and GS q

Mat getFrDctCa(int N, MatND a, MatND q)
//Reference: Cariolaro G, Erseghe T, Kraniauskas P(2002) The Fractional discrete
// cosine transform.IEEE Trans Signal Process 50(4) : 902 - 911
{
	int n, m, i, j, k;
	double phi, cos_wn[512], sin_wn[512];
	double r1, i1, r2, i2, A, B;
	double realEigenValues[1024], imagEigenValues[1024], temp[1024];
	Mat realEigenVectors = Mat(N, N, CV_64FC1), imagEigenVectors = Mat(N, N, CV_64FC1);
	char b[100]; _itoa(N, b, 10);

	if (N > 1024)
	{
		cout << endl << "N must not be larger than 1024";
		exit(0);
	}

	//读取特征值
	char fileName[100];
	strcpy(fileName, "Eigendata/EigenValues");
	strcat(fileName, b);
	strcat(fileName, ".dat");
	FILE* fp;
	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		cout << endl << "File " << fileName << " read error";
		exit(0);
	}
	fread(realEigenValues, sizeof(double), N, fp);
	fread(imagEigenValues, sizeof(double), N, fp);
	fclose(fp);

	//计算特征相角,只取(0,pi)之间的N/2个相角,0,2,4,...(2n)...,
	for (n = 0, k = 0; n < N / 2; n++, k += 2)
	{
		phi = atan2(imagEigenValues[k], realEigenValues[k]);
		cos_wn[n] = 2.0*cos((phi + 2.0*pi*q.at<double>(n))*a.at<double>(n));
		sin_wn[n] = 2.0*sin((phi + 2.0*pi*q.at<double>(n))*a.at<double>(n));
	}

	//读取特征向量
	strcpy(fileName, "Eigendata/EigenVectors");
	strcat(fileName, b);
	strcat(fileName, ".dat");
	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		cout << endl << "File " << fileName << " read error";
		exit(0);
	}

	//特征向量的文件格式是，特征向量1的实部,特征向量1的虚部,特征向量2的实部，特征向量2的虚部,......
	for (n = 0; n < N; n++)
	{
		fread(temp, sizeof(double), N, fp);
		for (m = 0; m < N; m++)
			realEigenVectors.at<double>(m, n) = temp[m];
		fread(temp, sizeof(double), N, fp);
		for (m = 0; m < N; m++)
			imagEigenVectors.at<double>(m, n) = temp[m];
	}
	fclose(fp);

	//计算FrDCT矩阵，分数阶a，Gs q,大小NxN
	Mat Ca = Mat::zeros(N, N, CV_64FC1);
	Mat tempCa = Mat(N, N, CV_64FC1);
	cout << endl;
	for (n = 0, k = 0; n < N / 2; n++, k += 2)
	{
		cout << "\rComputing Ca: " << n + 1 << " / " << N / 2;
		for (i = 0; i < N; i++)
		{
			r1 = realEigenVectors.at<double>(i, k);
			i1 = imagEigenVectors.at<double>(i, k);
			for (j = i; j < N; j++)
			{
				r2 = realEigenVectors.at<double>(j, k);
				i2 = imagEigenVectors.at<double>(j, k);
				A = (r1*r2 + i1*i2)*cos_wn[n];
				B = (r1*i2 - i1*r2)*sin_wn[n];
				tempCa.at<double>(i, j) = A + B;
				if (i != j)tempCa.at<double>(j, i) = A - B;
			}
		}
		add(Ca, tempCa, Ca);
	}
	return Ca;
}



// Function retuning DCT-II matrix
Mat dctMatrix(int N)
{
	Mat dctMtx = Mat(N, N, CV_64FC1);
	Mat e = Mat(1, N, CV_64FC1);//e = epesilon
	int k, n;
	double sqrt2 = sqrt(2), sqrtN = sqrt(N), pi = atan(1.0)*4.0;
	e.at<double>(0) = 1;
	if (N>1)
		for (int k = 1; k < N; k++)
			e.at<double>(k) = sqrt2;
	for (k = 0; k < N; k++)
		for (n = 0; n < N; n++)
			dctMtx.at<double>(k, n) = e.at<double>(k)*cos(pi*(n + 0.5)*k / N) / sqrtN;
	return dctMtx;
}

double getMSE(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_64F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2
	Scalar s = sum(s1);         // sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
	if (sse <= 1e-10) // for small values return zero
		sse = 1e-10;//return 0;
	double  mse = sse / (double)(I1.channels() * I1.total());
	return mse;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_64F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2
	Scalar s = sum(s1);         // sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
	if (sse <= 1e-15) // for small values return zero
		sse = 1e-15;//return 0;
	double  mse = sse / (double)(I1.channels() * I1.total());
	return 10 * log10(255 * 255.0 / mse);
}

//混沌序列的生成
Mat chaos(double seed, int ind, double u,int sizeMatrices)
{
	//ind - 截断起始位置
	//seed - 混沌系统的初始值
	//sizeMatrices - 序列的长度
	int lengthChaos;
	lengthChaos = sizeMatrices + ind;
	MatND temp = MatND::zeros(lengthChaos, 1, CV_64FC1);
	MatND F = MatND::zeros(sizeMatrices, 1, CV_64FC1);
	for (int i = 0; i < lengthChaos; i++)
	{
		temp.at<double>(i) = u*seed*(1 - seed);
		seed = temp.at<double>(i);
	}
	for (int j = ind,i=0; j < lengthChaos; j++, i++)
		F.at<double>(i) = temp.at<double>(j);
	return F;
}
Mat Twochaos(double seed1,double seed2,double u,int ind,int sizeMatrices)
{
	//ind - 截断起始位置
	//seed1 and seed2 - 混沌系统的两个初始值初始值
	//sizeMatrices - 序列的长度
	int lengthChaos;
	lengthChaos = sizeMatrices + ind;
	MatND temp1 = MatND::zeros(lengthChaos, 1, CV_64FC1);
	MatND temp2 = MatND::zeros(lengthChaos, 1, CV_64FC1);
	MatND F = MatND::zeros(2*sizeMatrices, 1, CV_64FC1);
	for (int i = 0; i < lengthChaos; i++)
	{
		temp1.at<double>(i) = u*seed1*(1 - seed1)+0.19*seed2*seed2;
		temp2.at<double>(i) = u*seed2*(1 - seed2) + 0.14*(seed1*seed1+seed1*seed2);
		seed1 = temp1.at<double>(i);
		seed2 = temp2.at<double>(i);
	}
	for (int j = ind, i = 0; j < lengthChaos; j++, i++)
	{
		F.at<double>(i) = temp1.at<double>(j);
		F.at<double>(j+ sizeMatrices)= temp2.at<double>(j);
	}
	return F;
}

//三维超混沌序列的生成
Mat threeDchaoticmap(double x0, double y0, double z0, double a, double b, double l)
{
	int N =70001 ,image_height=512;
	MatND x = Mat::zeros(N, 1, CV_64FC1);   
	MatND y = Mat::zeros(N, 1, CV_64FC1);
	MatND z = Mat::zeros(N, 1, CV_64FC1);  
	MatND f = Mat::zeros(3*N, 1, CV_64FC1);

	x.at<double>(0) = x0;
	y.at<double>(0) = y0;
	z.at<double>(0) = z0;
	for (int i = 1; i < N; i++) {
		//x序列的获得
		x.at<double>(i) = l* x.at<double>(i-1)*(1- x.at<double>(i-1))
			+b*y.at<double>(i-1)*y.at<double>(i-1)*x.at<double>(i-1)
			+a*z.at<double>(i-1)*z.at<double>(i-1)*z.at<double>(i-1);
		//y序列的获得
		y.at<double>(i) = l * y.at<double>(i-1)*(1 - y.at<double>(i-1))
			+ b*z.at<double>(i-1)*z.at<double>(i-1)*y.at<double>(i-1)
			+ a*x.at<double>(i-1)*x.at<double>(i-1)*x.at<double>(i-1);
		//z序列的获得
		z.at<double>(i)=l* z.at<double>(i-1)*(1- z.at<double>(i-1))
			+ b*x.at<double>(i-1)*x.at<double>(i-1)*z.at<double>(i-1)
			+ a*y.at<double>(i-1)*y.at<double>(i-1)*y.at<double>(i-1);
	}
	//以下程序是求余和取整
	for (int i = 0; i < N; i++) {
		x.at<double>(i) = ceil(fmod(x.at<double>(i)* 100000, image_height));
		y.at<double>(i) = ceil(fmod(y.at<double>(i) * 100000, image_height));
		z.at<double>(i) = ceil(fmod(z.at<double>(i) * 100000, image_height));
	}

	for (int i = 0; i < 3*N; i++) {
		if (i < N)
			f.at<double>(i) = x.at<double>(i);
		else if(i>=N && i<2*N)
			f.at<double>(i) = y.at<double>(i-N);
		else
			f.at<double>(i) = z.at<double>(i-2*N);
	}
	return f;
}

//第一种置乱方案：基于3维超混沌系统的行列搬移(顺过程)
Mat rotationscrambling(Mat &I1, MatND x, MatND y, MatND z) 
{
	int row = I1.rows;
	int col = I1.cols;
	int n = 5000;
	int p = 6000;
	int q = 700;

	MatND K = Mat::zeros(row, 1, CV_64FC1);   
	MatND I = Mat::zeros(row, 1, CV_64FC1);
	MatND M= Mat::zeros(row*col, 1, CV_64FC1);  
	Mat sh_row = Mat::zeros(row, col, CV_64FC1);
	Mat sh_col = Mat::zeros(row, col, CV_64FC1);

	for (int j = 0; j < row; j++) {
		K.at<double>(j) = x.at<double>(j + n);
		I.at<double>(j) = y.at<double>(j + p);
	}
	cout << K.at<double>(65);
	cout << I1.at<double>(0, 65);
	//cout << I;
	/*for (int j = 0; j < row*col; j++) {
		M.at<double>(j) = z.at<double>(j + q);
	}*/
	
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (fmod(K.at<double>(i), 2) == 0) {
				if ((j + K.at<double>(i)) <= col)
					sh_row.at<double>(i, j + K.at<double>(i)) = I1.at<double>(i, j);
				else
					sh_row.at<double>(i, j + K.at<double>(i)-col) = I1.at<double>(i, j);
			}
			else {
				if ((j - K.at<double>(i)) >= 1)
					sh_row.at<double>(i, j - K.at<double>(i)) = I1.at<double>(i, j);
				else
					sh_row.at<double>(i, (col+j - K.at<double>(i))) = I1.at<double>(i, j);
			}
		}
	}

	for (int j = 0; j < col; j++) {
		for (int i = 0; i < row; i++) {
			if (fmod(I.at<double>(j), 2) == 0) {
				if ((i - I.at<double>(j)) >= 1)
					sh_col.at<double>(i - I.at<double>(j), j) = sh_row.at<double>(i, j);
				else
					sh_col.at<double>(row+ i - I.at<double>(j), j ) = sh_row.at<double>(i, j);
			}
			else {
				if ((i + I.at<double>(j))<=row)
					sh_col.at<double>(i + I.at<double>(j), j ) = sh_row.at<double>(i, j);
				else
					sh_col.at<double>(i+I.at<double>(j)-row, j ) = sh_row.at<double>(i, j);
			}
		}
	}
	return sh_col;
}
//第一种置乱方案：基于3维超混沌系统的行列搬移(逆过程)
Mat inverserotationscrambling(Mat &I1, MatND x, MatND y, MatND z) {
	int row = I1.rows;
	int col = I1.cols;
	int n = 500;
	int p = 600;
	int q = 700;

	MatND K = Mat::zeros(row, 1, CV_64FC1);
	MatND I = Mat::zeros(row, 1, CV_64FC1);
	MatND M = Mat::zeros(row*col, 1, CV_64FC1);
	Mat sh_row = Mat::zeros(row, col, CV_64FC1);
	Mat sh_col = Mat::zeros(row, col, CV_64FC1);

	for (int j = 0; j < row; j++) {
		K.at<int>(j) = x.at<int>(j + n);
		I.at<int>(j) = y.at<int>(j + p);
	}
	for (int j = 0; j < row*col; j++) {
		M.at<int>(j) = z.at<int>(j + q);
	}
	for (int j = 0; j < col; j++) {
		for (int i = 0; i < row; i++) {
			if (fmod(I.at<int>(j), 2) == 0) {
				if ((i + I.at<int>(j)) <= row)
					sh_col.at<double>(i + I.at<int>(j), j) = I1.at<double>(i, j);
				else
					sh_col.at<double>(i + I.at<int>(j) - row, j) = I1.at<double>(i, j);
				
			}
			else {
				if ((i - I.at<int>(j)) >= 1)
					sh_col.at<double>(i - I.at<int>(j), j) = I1.at<double>(i, j);
				else
					sh_col.at<double>(row + i - I.at<int>(j), j) = I1.at<double>(i, j);
			}
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (fmod(K.at<int>(i), 2) == 0) {
				if ((j - K.at<int>(i)) >= 1)
					sh_row.at<double>(i, j - K.at<int>(i)) = sh_col.at<double>(i, j);
				else
					sh_row.at<double>(i, col + j - K.at<int>(i)) = sh_col.at<double>(i, j);
				
			}
			else {
				if ((j + K.at<int>(i)) <= col)
					sh_row.at<double>(i, j + K.at<int>(i)) = sh_col.at<double>(i, j);
				else
					sh_row.at<double>(i, j + K.at<int>(i) - col) = sh_col.at<double>(i, j);
			}
		}
	}
	return sh_row;
}

//第二种置乱方案猫映射
Mat Arnold(Mat& I1,Mat& C) {
	int row = I1.rows;
	int col = I1.cols;
	Mat cipher= Mat::zeros(row, col, CV_64FC1);
	Mat fcipher = Mat::zeros(row, col, CV_64FC1);
	//int key1 = 5, key2 = 2, key3 = 1, key4 = 1, key5 = 1;
	int key1=5,key2=5,key3=2,key4=7,key5=3;
	int x, y;

	for(int k=0;k<key1;k++) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			x = mod(key2*i + key3*j, row) ;
			y = mod((key4*i + key5*j), row);
			cipher.at<double>(x, y) = I1.at<double>(i, j);
			//cout << (i, j) << "--" << (x, y);
		}
	  }
	}

	return cipher;
}

//图像矩阵的异或操作
Mat imageXor(Mat& I1, Mat &C) {
	int row = I1.rows;
	int col = I1.cols;
	Mat cipher = Mat::zeros(row, col, CV_64FC1);
	Mat fcipher = Mat::zeros(row, col, CV_64FC1);

	//bitwise_xor(I1,C1,cipher);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			fcipher.at<double>(i, j) = static_cast<int>(I1.at<double>(i, j))^(static_cast<int>(C.at<double>(i,j)));
		}
	}
	cout <<"image not xor="<<I1.at<double>(20, 50) << endl;
	cout << "最终加密图的像素值="<<fcipher.at<double>(20, 50)<<endl;
	return fcipher;
}
//图像矩阵的异或操作(逆操作)
Mat inverseimageXor(Mat& I1, Mat &C) {
	int row = I1.rows;
	int col = I1.cols;
	Mat cipher = Mat::zeros(row, col, CV_64FC1);
	Mat fcipher = Mat::zeros(row, col, CV_64FC1);

	//bitwise_xor(I1,C1,cipher);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			fcipher.at<double>(i, j) = static_cast<int>(I1.at<double>(i, j)) ^ (static_cast<int>(C.at<double>(i, j)));
		}
	}
	cout <<"image xored=" <<fcipher.at<double>(20, 50) << endl;
	return fcipher;
}

//量化操作
Mat lianghua(Mat& I1, double Mx, double Mn) {
	int M = I1.rows;
	int N = I1.cols;
	Mat H = Mat::zeros(M,N, CV_64FC1);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			H.at<double>(i, j) = (I1.at<double>(i, j) - Mn) / (Mx-Mn);
			H.at<double>(i, j) = round(double(H.at<double>(i, j) * 255));   //量化求余过程
			//H.at<double>(i, j) = double(H.at<double>(i, j) * 255);
		}
	}
	return H;
}
//逆量化操作
Mat ilianghua(Mat& I1, double Mx, double Mn) {
	int M = I1.rows;
	int N = I1.cols;
	Mat H = Mat::zeros(M, N, CV_64FC1);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			I1.at<double>(i, j) = (I1.at<double>(i, j) / (255));     //量化求余过程
			//I1.at<double>(i, j) = double(I1.at<double>(i, j)/double(255));
			H.at<double>(i, j) = I1.at<double>(i, j) * (Mx-Mn)+Mn;
		}
	}
	return H;
}


//第二种置乱方案猫映射(逆映射)
Mat iArnold(Mat& I1, Mat &C) {
	int row = I1.rows;
	int col = I1.cols;
	Mat cipher = Mat::zeros(row, col, CV_64FC1);
	Mat fcipher = Mat::zeros(row, col, CV_64FC1);
	int x, y;
	//int key1 = 5, key2 = 3, key3 = -2, key4 = -7, key5 = 5;
	int key1=5,key2 = 5, key3 = 2, key4 = 7, key5 = 3;

	//int key2 = 2, key3 = 1, key4 = 1, key5 = 1;
	for (int k = 0 ; k <key1; k++) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			x = mod(key2*i + key3*j, row);
			y = mod((key4*i + key5*j), row);
			cipher.at<double>(i, j) = I1.at<double>(x, y);
		}
	}
	}
	/*for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			x = fmod(i + pd*j, row - 1) + 1;
			y = fmod((qd*i + (pd*qd + 1)*j), row - 1) + 1;
			cipher.at<double>(i, j) = I1.at<double>(x, y);
		}
	}*/
	return cipher;
}

//生命游戏置乱
Mat GameOflife(Mat J, int h, Mat a) {
	int row_size = J.rows;
	int cols_size = J.cols;
	int alpha_size = h;
	typedef Vec<double, 5> Vec5d;
	//生成一个512x512x15的Mat，数据为double型
	Mat M = Mat::zeros(row_size, cols_size, CV_64FC(h));
	cout << "channel = " << M.channels() << endl;//输出为5
	Mat temp1 = Mat::zeros(row_size + 2, cols_size + 2, CV_64FC1);
	Mat temp2 = Mat::zeros(3, 3, CV_64FC1);
	for (int c = 0; c < M.channels(); c++)
	{
		//循环边界条件
		for (int i = 0; i < M.rows-2; i++) {
			for (int j = 0; j < M.cols-2; j++)
			{
				temp1.at<double>(i + 1, j + 1) = a.at<double>(i, j);
			}
		}
		temp1.at<double>(0, 0) = a.at<double>(511, 511);
		temp1.at<double>(0, 511) = a.at<double>(511, 0);
		temp1.at<double>(511, 0) = a.at<double>(0, 511);
		temp1.at<double>(511, 511) = a.at<double>(0, 0);
	
		for (int i = 1; i < M.rows+1; i++)
		{
			for (int j = 1; j < M.cols+1; j++)
			{  
				Mat row = temp1.rowRange(i-1,i+1);
				temp2 = row.colRange(j - 1, j + 1);
				//根据生命游戏规，取矩阵a元素（x，y）周围八个元素值的和，因为最外边一圈元素周围没有八个相邻元素，
				//所以不考虑，x、y范围2:m - 1、2 : n - 1。
				Scalar temp=sum(temp2);
				cout << temp[0] << endl;

				//如果（x，y）周围存在2个1，即游戏意义2个活细胞，则这个细胞下一刻生死与原来生死有关。
				if (temp[0] == 2)
					break;
				//如果（x，y）周围存在3个1，即游戏意义2个活细胞，则这个细胞下一刻必存活（这里我假设1是存活）。
				else if (temp[0] == 3)
					a.at<double>(i - 1, j - 1) = 1;
				//如果（x，y）周围存在活细胞不是这两个值，即太多或太少，则这个细胞下一刻必死亡（这里我假设0是死亡）。
				else
					a.at<double>(i - 1, j - 1) = 0;

				M.at<Vec5d>(i, j)[c] = a.at<double>(i,j);
			}
		}
	}
	Mat img;
	transpose(J,img);
	MatND tempimg = img.reshape(row_size*cols_size, 1);
	int num = 0;

	//encryption
	//step3(row)
	Mat result = Mat::zeros(row_size, cols_size, CV_64FC1);
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++)
		{
			a.at<double>(i,j) = M.at<Vec5d>(i, j)[0];
			if (a.at<double>(i, j))
				result.at<double>(i,j)=img.at<double>(num);
			num = num + 1;
		}
	}

	//step4(row)
	for (int q = 1; q < h; q++) {
		Mat temp3 = Mat::zeros(row_size, cols_size, CV_64FC1);
		for (int index = 0; index < q - 1; index++) {
			for (int i = 0; i < M.rows; i++) {
				for (int j = 0; j < M.cols; j++)
				{
					temp3.at<double>(i, j) = temp3.at<double>(i, j) + M.at<Vec5d>(i, j)[index];
				}
			}
		}
		for (int i = 0; i < M.rows; i++) {
			for (int j = 0; j < M.cols; j++)
			{
				if (a.at<double>(i, j) && !temp3.at<double>(i, j))
					result.at<double>(i, j) = img.at<double>(num);
				num = num + 1;
			}
		}
	}

	//step5(row)
	Mat temp4 = Mat::zeros(row_size, cols_size, CV_64FC1);
	for (int q = 0; q < h; q++) {
			for (int i = 0; i < M.rows; i++) {
				for (int j = 0; j < M.cols; j++)
				{
					temp4.at<double>(i, j) = temp4.at<double>(i, j) + M.at<Vec5d>(i, j)[q];
				}
			}
		}
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++)
		{
			if ( !temp4.at<double>(i, j))
				result.at<double>(i, j) = img.at<double>(num);
			num = num + 1;
		}
	}

	Mat TEMP = result;

	//step3(col)
	Mat fresult = Mat::zeros(row_size, cols_size, CV_64FC1);
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++)
		{
			a.at<double>(i, j) = M.at<Vec5d>(i, j)[0];
			if (a.at<double>(i, j))
				fresult.at<double>(i, j) = TEMP.at<double>(num);
			num = num + 1;
		}
	}

	//step4(col)
	for (int q = 1; q < h; q++) {
		Mat temp5 = Mat::zeros(row_size, cols_size, CV_64FC1);
		for (int index = 0; index < q - 1; index++) {
			for (int i = 0; i < M.rows; i++) {
				for (int j = 0; j < M.cols; j++)
				{
					temp5.at<double>(i, j) = temp5.at<double>(i, j) + M.at<Vec5d>(i, j)[index];
				}
			}
		}
		for (int i = 0; i < M.rows; i++) {
			for (int j = 0; j < M.cols; j++)
			{
				if (a.at<double>(i, j) && !temp5.at<double>(i, j))
					fresult.at<double>(i, j) = TEMP.at<double>(num);
				num = num + 1;
			}
		}
	}

	//step5(col)
	Mat temp6 = Mat::zeros(row_size, cols_size, CV_64FC1);
	for (int q = 0; q < h; q++) {
		for (int i = 0; i < M.rows; i++) {
			for (int j = 0; j < M.cols; j++)
			{
				temp6.at<double>(i, j) = temp6.at<double>(i, j) + M.at<Vec5d>(i, j)[q];
			}
		}
	}
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++)
		{
			if (!temp6.at<double>(i, j))
				fresult.at<double>(i, j) = TEMP.at<double>(num);
			num = num + 1;
		}
	}

	return fresult;
}
//求图像的信息熵
double Entropy(Mat img)
{
	//将输入的矩阵为图像
	double temp[256];
	for (int i = 0; i<256; i++)
	{
		temp[i] = 0.0;
	}
	//计算每个像素的累积值
	for (int m = 0; m<img.rows; m++)
	{
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n<img.cols; n++)
		{
			int i = t[n];
			temp[i] = temp[i] + 1;
		}
	}
	//计算每个像素的概率
	for (int i = 0; i<256; i++)
	{
		temp[i] = temp[i] / (img.rows*img.cols);
	}
	double result = 0;
	//根据定义计算图像熵
	for (int i = 0; i<256; i++)
	{
		if (temp[i] == 0.0)
			result = result;
		else
			result = result - temp[i] * (log(temp[i]) / log(2.0));
	}
	return result;
} 

/*
//判断是否为素数
bool isPrime(unsigned long n) {
	if (n == 2 || n == 3)
		return true;
	if (n % 6 != 1 && n % 6 != 5)//素数聚集原理
		return false;
	float nsqrt = floor(sqrt((float)n));
	//若小于nsqrt的数中不存在因数则大于nsqrt的数中也不会存在其因数。
	//n = 6*x + 1 or 6*x -1，它非2、3和6的倍数。
	for (int i = 6; i <= nsqrt; i += 6) {
		if (n % (i - 1) == 0 || n % (i + 1) == 0)
			return false;
	}
	return true;
}

MatND factor(int N)
{
	int n, i;
	int count = 0;
	n = N;
	MatND d;
	for (i = 2; i <= n; i++) {
		while (n != i) {
			if (n%i == 0) {
				count = count + 1;
				d.at<double>(count) = i;
				n = n / i;
			}
			else {
				break;
			}
		}
	}
	return d;
}

MatND sushu(int N) 
{
	int j = 1; 
	//求1到N的质数
	MatND px;
	for (int i = 0; i++; i < N)
	{
		if (isPrime(i)) {
			px.at<double>(j) = i;
			j = j + 1;
		}
	}
	MatND a = factor(N);
	int m = a.row;
	//因式分解后的因子存入b,且不重复。
	MatND b;
	b.at<double>(1) = a.at<double>(1);
	for (int i = 2; i <= m; i++)
	{
		if (a.at<double>(i - 1) == a.at<double>(i))
			b.at<double>(j) = b.at<double>(j);
		else
		{
			j = j + 1;
			b.at<double>(j) = a.at<double>(i);
		}
	}
	//互素的数提取出来
	MatND c;
	int z = 1;
	int m1 = b.row;
	MatND y = px;
	int n1 = y.row;
	for (int i = 0; i < m1; i++)
	{
		for (int j = 0; j < n1; j++)
		{
			//if()
		}
	}
}
*/

/*Mat logistic2(double seed1, double seed2, int ind, double u1,double u2,int sizeMatrices)
{
	//ind - 截断起始位置
	//seed - 混沌系统的初始值
	//sizeMatrices - 序列的长度
	int lengthChaos;
	lengthChaos = sizeMatrices + ind;		
}
*/

/*
Mat dctMatrix(int N);
Mat frDct(Mat& I,double a, MatND q);
Mat getFrDctCa(int N,double a,MatND q);
double getMSE(const Mat& I1, const Mat& I2);
double getPSNR(const Mat& I1, const Mat& I2);
Mat chaos(double seed, int  sizeMatrices,int ind);
double pi = atan(1.0)*4.0;

//Function returning fractional DCT of an image with fraction a and GS q

Mat frDct(Mat& I,double a,MatND q)
{
	int i, j, M, N;
	clock_t duration;
	M = I.rows;
	N = I.cols;
	if (M != N)
	{
		cout << endl << " Height and width of the image must agree, exit";
		exit(0);
	}
	Mat J = Mat(M, N, CV_64FC1);
	duration = clock();
	Mat Ca = getFrDctCa(N, a, q);
	duration = (clock() - duration)/CLOCKS_PER_SEC;
	cout << endl << duration << " seconds for getting a FrDCT matrix of size " << M << " x " << N;
	//行行变换
	MatND temp = Mat(N, 1, CV_64FC1);
	for (i = 0;i < M;i++)
	{
		for (j = 0;j < N;j++)
			temp.at<double>(j) = I.at<double>(i, j);
		temp = Ca*temp;
		for (j = 0;j < N;j++)
			J.at<double>(i, j) = temp.at<double>(j);
	}
	//列列变换
	temp = Mat(M, 1, CV_64FC1);
	for (j = 0;j < N;j++)
	{
		for (i = 0;i < M;i++)
			temp.at<double>(i) = J.at<double>(i, j);
		temp = Ca*temp;
		for (i = 0;i < M;i++)
			J.at<double>(i, j) = temp.at<double>(i);
	}
	return J;
}

//Function returning fractional DCT matrix with fraction a and GS q
Mat getFrDctCa(int N, double a, MatND q)
//Reference: Cariolaro G, Erseghe T, Kraniauskas P(2002) The Fractional discrete
// cosine transform.IEEE Trans Signal Process 50(4) : 902 - 911
{
	int n, m, i, j, k;
	double phi, cos_wn[512], sin_wn[512];
	double r1, i1, r2, i2, A, B;
	double realEigenValues[1024], imagEigenValues[1024], temp[1024];
	Mat realEigenVectors = Mat(N, N, CV_64FC1),imagEigenVectors = Mat(N, N, CV_64FC1);
	char b[100];_itoa(N, b, 10);

	if (N > 1024)
	{
		cout << endl << "N must not be larger than 1024";
		exit(0);
	}
	
	//读取特征值
	char fileName[100];
	strcpy(fileName,"Eigendata/EigenValues");
	strcat(fileName, b);
	strcat(fileName, ".dat");
	FILE* fp;
	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		cout << endl << "File " << fileName << " read error";
		exit(0);
	}
	fread(realEigenValues, sizeof(double), N, fp);
	fread(imagEigenValues, sizeof(double), N, fp);
	fclose(fp);

	//计算特征相角,只取(0,pi)之间的N/2个相角,0,2,4,...(2n)...,
	for (n = 0,k = 0;n < N/2; n++,k+=2)
	{
		phi = atan2(imagEigenValues[k], realEigenValues[k]);
		cos_wn[n] = 2.0*cos((phi + 2.0*pi*q.at<double>(n))*a);
		sin_wn[n] = 2.0*sin((phi + 2.0*pi*q.at<double>(n))*a);
	}

	//读取特征向量
	strcpy(fileName, "Eigendata/EigenVectors");
	strcat(fileName, b);
	strcat(fileName, ".dat");
	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		cout << endl << "File " << fileName << " read error";
		exit(0);
	}

	//特征向量的文件格式是，特征向量1的实部,特征向量1的虚部,特征向量2的实部，特征向量2的虚部,......
	for (n = 0;n < N;n++)
	{
		fread(temp, sizeof(double), N, fp);
		for (m = 0;m < N;m++)
			realEigenVectors.at<double>(m, n) = temp[m];
		fread(temp, sizeof(double), N, fp);
		for (m = 0;m < N;m++)
			imagEigenVectors.at<double>(m, n) = temp[m];
	}
	fclose(fp);
	
	//计算FrDCT矩阵，分数阶a，Gs q,大小NxN
	Mat Ca = Mat::zeros(N, N, CV_64FC1);
	Mat tempCa = Mat(N, N, CV_64FC1);
	cout << endl;
	for (n = 0, k = 0;n < N/2;n++,k+=2)
	{		
		cout << "\rComputing Ca: " << n+1 <<" / "<<N/2;
		for (i = 0;i < N;i++)
		{
			r1 = realEigenVectors.at<double>(i, k);
			i1 = imagEigenVectors.at<double>(i, k);
			for (j = i;j < N;j++)
			{
				r2 = realEigenVectors.at<double>(j, k);
				i2 = imagEigenVectors.at<double>(j, k);
				A = (r1*r2 + i1*i2)*cos_wn[n];
				B = (r1*i2 - i1*r2)*sin_wn[n];
				tempCa.at<double>(i, j) = A + B;
				if (i != j)tempCa.at<double>(j, i) = A - B;
			}
		}
		add(Ca, tempCa, Ca);
	}	
	return Ca;
}

// Function retuning DCT-II matrix
Mat dctMatrix(int N)
{
	Mat dctMtx = Mat(N, N, CV_64FC1);
	Mat e = Mat(1, N, CV_64FC1);//e = epesilon
	int k, n;
	double sqrt2 = sqrt(2), sqrtN = sqrt(N), pi = atan(1.0)*4.0;
	e.at<double>(0) = 1;
	if(N>1)
		for (int k = 1;k < N; k++)
			e.at<double>(k) = sqrt2;
	for (k = 0; k < N; k++)
		for (n = 0; n < N; n++)
			dctMtx.at<double>(k, n) = e.at<double>(k)*cos(pi*(n + 0.5)*k / N ) / sqrtN;
	return dctMtx;
}

double getMSE(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_64F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2
	Scalar s = sum(s1);         // sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
	if (sse <= 1e-10) // for small values return zero
		sse = 1e-10;//return 0;
	double  mse = sse / (double)(I1.channels() * I1.total());
	return mse;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_64F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2
	Scalar s = sum(s1);         // sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
	if (sse <= 1e-15) // for small values return zero
		sse = 1e-15;//return 0;
	double  mse = sse / (double)(I1.channels() * I1.total());
	return 10*log10(255*255.0/mse);
}

//混沌序列的生成
Mat chaos(double seed, int ind, double u,int sizeMatrices)
{
//ind - 截断起始位置
//seed - 混沌系统的初始值
//sizeMatrices - 序列的长度
int lengthChaos;
lengthChaos = sizeMatrices + ind;
MatND temp = MatND::zeros(lengthChaos, 1, CV_64FC1);
MatND F = MatND::zeros(sizeMatrices, 1, CV_64FC1);
for (int i = 0; i < lengthChaos; i++)
{
temp.at<double>(i) = u*seed*(1 - seed);
seed = temp.at<double>(i);
}
for (int j = ind,i=0; j < lengthChaos; j++, i++)
F.at<double>(i) = temp.at<double>(j);
return F;
}



Mat chaos(double seed, int sizeMatrices, int ind)
{
	//ind - 截断起始位置
	//seed - 混沌系统的初始值
	//sizeMatrices - 序列的长度
	int lengthChaos;
	lengthChaos = sizeMatrices + ind;
	Mat temp = Mat::zeros(1, lengthChaos, CV_64FC1);
	Mat F = Mat::zeros(1, sizeMatrices, CV_64FC1);
	for (int i = 0; i < lengthChaos; i++)
	{
		temp.at<double>(i) = 3.99999*seed*(1 - seed);
		seed = temp.at<double>(i);
	}
	for (int j = ind; j < lengthChaos; j++)
		F.at<double>(j-ind) = temp.at<double>(j);
	return F;
}
*/
#endif