#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "plot.h"//包含plot.h是为了绘图
#include "frdct.h"//包含frdct.h是为了调用保实分数DCT，调用方法见frdct.h

using namespace std;
using namespace cv;




void main()
{
	char* file = "lena.bmp";
	int i=0, M, N;
	Mat picture = imread(file, IMREAD_GRAYSCALE);//强行转换为灰度图像
	if (!picture.data)
	{
		printf("\nFile %s read error\n", file);
		system("pause");
		return;
	}
	else
		printf("\nFile %s read successfully\n", file);

	//cout << picture.at<double>(1, 1) << endl;
	//先转换下格式  
	Mat image, frDctImage,frDctImageRec,logFrDctImage;
	picture.convertTo(image, CV_64FC1);

	M = image.rows;
	N = image.cols;

	//密钥
	double u = 3.999, seed = 0.2;
	int ind = 1000;
	
	/*MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N / 2, 1, CV_64FC1);   //分数阶向量化
	MatND q = Mat::zeros(N / 2, 1, CV_64FC1);  //生成序列设为0，长度 N/2
	for (int i = 0; i < N / 2; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i + N / 2));
	}*/

	//分数阶向量和GS第一种情况
	//MatND t = chaos(seed, ind, u, 2 * N);
	//MatND a = Mat::zeros(N, 1, CV_64FC1);   //分数阶向量化
	//MatND b= Mat::zeros(N, 1, CV_64FC1);
	//MatND q = Mat::zeros(N, 1, CV_64FC1);  //生成序列设为0，长度 N/2
	//for (int i = 0; i < N; i++)
	//{
	//	a.at<double>(i) = t.at<double>(i);
	//	q.at<double>(i) = round(t.at<double>(i + N));
	//	b.at<double>(i) = 0.25;
	//}


	//分数阶向量和GS第二种情况
	MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N, 1, CV_64FC1);   //分数阶向量化
	MatND b = Mat::zeros(N, 1, CV_64FC1);
	MatND q = Mat::zeros(N, 1, CV_64FC1);  //生成序列设为0，长度 N/2
	double k = 800.0 / 701.0;
	double MM = 150.0;
	int count = 1;
	int j = 1;
	while (count != N) {
			if (isrp(j, int(MM))) {
				a.at<double>(i) = double(j )/MM;
				count++;
				i++;
			}
			j++;
			}
	for (int i = 0; i < N; i++) {
		b.at<double>(i) = 0.25;
		q.at<double>(i) = round(t.at<double>(i));
	}
	////置乱方案一：行列搬移
	//double x0 = 0.2350,y0=0.3500,z0=0.7350,a0=0.0125,b0=0.0157,r=3.7700;
	//MatND F= threeDchaoticmap(x0,y0,z0,a0,b0,r);
	//int times = 70001;
	//MatND x = Mat::zeros(times, 1, CV_64FC1);   //分数阶向量化
	//MatND y = Mat::zeros(times, 1, CV_64FC1);
	//MatND z = Mat::zeros(times, 1, CV_64FC1);
	////MatND f = Mat::zeros(N, 1, CV_64FC1);
	//for (int i = 0; i < times; i++) {
	//	x.at<double>(i) = F.at<double>(i);
	//	y.at<double>(i) = F.at<double>(i + times);
	//	z.at<double>(i) = F.at<double>(i + 2*times);
	//}
	//Mat scrambleimage;
	//scrambleimage = rotationscrambling(image, x, y, z);

	//置乱方案二：猫映射
	//double key1 = 225, qd = 153;
	MatND bix = chaos(seed, ind, u, N*N);
	for (int i = 0; i < N*N; i++) {
		bix.at<double>(i) = round(fmod(bix.at<double>(i) * 10000, 256));
	}
	cout << bix.at<double>(5)<<endl;
	Mat C= bix.reshape(N,N);
	cout << C.at<double>(120, 200)<<endl;
	

	frDctImage = frDct(image, a, q);
	//以下程序是求图像的像素值的最大和最小值
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(frDctImage, minp, maxp);
	cout << "Mat minv = " << minv << endl;
	cout << "Mat maxv = " << maxv << endl;
	
	//量化过程
	Mat lfrDctImage, ilfrDctImage;
	lfrDctImage = lianghua(frDctImage, maxv, minv);
	cout << "fr="<<frDctImage.at<double>(20, 50) << endl;
	
	Mat scrambleimage,rscrambleimage;
	Mat iscrambleimage;
	Mat finalencryptedimage, ifinalencryptedimage;
	scrambleimage = Arnold(lfrDctImage, C);    //猫映射


	finalencryptedimage = imageXor(scrambleimage,C);  //图像异或操作
	
	//最终加密图像的显示
	Mat logFinalencryptedimage;
	logFinalencryptedimage= finalencryptedimage;
	//log(abs(finalencryptedimage) + 1, logFinalencryptedimage);    //取对数，背景亮白
	normalize(logFinalencryptedimage, logFinalencryptedimage, 0, 255, CV_MINMAX, CV_8UC1);//归一化像素值[0,255] 
	string windowFinal = "final encryptedimage";
	namedWindow(windowFinal, WINDOW_NORMAL);
	imshow(windowFinal, logFinalencryptedimage);

	ifinalencryptedimage = inverseimageXor(finalencryptedimage, C);     //逆异或操作
	//iscrambleimage = iArnold(scrambleimage, C);           //逆映射    如果量化过程没有求余，图像解密效果非常好，PSNR很大230.329
	iscrambleimage = iArnold(ifinalencryptedimage, C);

	//Mat scrambleimage;
	//scrambleimage = Arnold(frDctImage, C, pd, qd);
	////方案2显示置乱后的图像
	//Mat logScrambleimage;
	//log(abs(scrambleimage)+1, logFrDctImage);
	//normalize(logScrambleimage , logScrambleimage, 0, 255, CV_MINMAX, CV_8UC1);
	//string windowScram = "Scramble image";
	//namedWindow(windowScram, WINDOW_NORMAL);
	//imshow(windowScram, scrambleimage);


	ilfrDctImage = ilianghua(iscrambleimage, maxv, minv);
	cout << "fr=" << ilfrDctImage.at<double>(20, 50) << endl;
	frDctImageRec = frDct(ilfrDctImage, -a, q);
	cout << "\nPSNR= " << getPSNR(frDctImageRec, image) << " dB";
   
	//log(abs(scrambleimage)+1, logFrDctImage);
	normalize(scrambleimage, scrambleimage, 0, 255, CV_MINMAX, CV_8UC1);//归一化像素值[0,255] 
	normalize(frDctImageRec, frDctImageRec, 0, 255, CV_MINMAX, CV_8UC1);

	string windowOri = "Image original";
	string windowTrans = "Image FrDCT";
	string windowRec = "Image reconstruction";
	namedWindow(windowOri, WINDOW_NORMAL);
	imshow(windowOri, picture);

	namedWindow(windowTrans, WINDOW_NORMAL);
	imshow(windowTrans, scrambleimage);
	namedWindow(windowRec, WINDOW_NORMAL);
	imshow(windowRec, frDctImageRec);


	//下面开始调试FrDCT,上面的Lena图像显示和下面的程序没有什么关系

	N = 16; //信号长度
	MatND signal1 = Mat::zeros(N, 1, CV_64FC1);

	//通过取舍注释，选取信号
	for (i = 0;i < 3;i++)signal1.at<double>(i) = 1; //方波信号，3个1，13个0
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = cos(2 * pi*(i + 1) / 16);//余弦信号
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = (0.005*exp(i / 2.0) + 1)/11.0;//指数信号
	//cout << endl << "signal1 = " << signal1;
	//为了求分数DCT，先取得分数变换矩阵，这样在重复调用时，可以直接利用这个矩阵，节约时间
	Mat Ca;
	Ca = getFrDctCa(N, a, q);  //变换矩阵，分数阶a
	Mat Cb;
	Cb = getFrDctCa(N, b, q);  //变换矩阵，分数阶b
	Mat Ca_;
	Ca_ = getFrDctCa(N, -a, q);//逆变换矩阵，分数阶a
	Mat Cb_;
	Cb_ = getFrDctCa(N, -b, q);//逆变换矩阵，分数阶b
	
	Mat frDctSignal1 = Ca*signal1; //再做变换
	//cout << "\nFrDCT signal1 = \n" << frDctSignal1;//打印变换结果
	Mat frDctSignal1Rec = Ca_*frDctSignal1;//逆变换
	//cout << "\nsignal1 = \n" << signal1;//对比结果,验证分数阶为负数，即是逆变换
	//cout << "\nReconstruction from FrDCT of signal1\n" << frDctSignal1Rec;//打印变换结果
	double mse1 = getMSE(signal1, frDctSignal1Rec);//求均方误差
	cout << endl << "Mse1 = " << mse1;//如果mse约等于0，逆变换OK

	Mat Cab;
	Cab = getFrDctCa(N, a + b, q);//变换矩阵，分数阶a+b
	Mat frDctSignal2 = Cb*frDctSignal1;//Ca的结果上做Cb
	Mat frDctSignal3 = Cab*signal1;//直接做Cab，验证分数阶可加性
	double mse2 = getMSE(frDctSignal2, frDctSignal3);//求均方误差
	cout << endl << "Mse2 = " << mse2;//如果mse约等于0，满足可加性

	Mat C1;
	MatND c = Mat::zeros(N, 1, CV_64FC1);
	for (int i = 0; i < N; i++)
		c.at<double>(i) = i;
	C1 = getFrDctCa(N, c, q);
	Mat dctSignal;
	dct(signal1, dctSignal);//普通DCT
	double mse3 = getMSE(C1*signal1, dctSignal);//和分数阶为1时的FrDCT比较
	cout << endl << "Mse3 = " << mse3;//如果mse约等于0，证明分数阶为1时，即是常规dct

	Plot plot1;
	plot1.plot(signal1, CvScalar(255, 0, 0), 'd');
	namedWindow("Signal 1", WINDOW_NORMAL);
	cvShowImage("Signal 1", plot1.Figure);//原始信号plot
	
	Plot plot2;
	plot2.plot(frDctSignal1, CvScalar(0, 255, 0), 'd');
	namedWindow("FrDct of Signal 1", WINDOW_NORMAL);
	cvShowImage("FrDct of Signal 1", plot2.Figure);//原始信号的FrDCT的Plot

	Plot plot3;
	plot3.plot(frDctSignal1Rec, CvScalar(0, 0, 255), 'd');
	namedWindow("Reconstruction from FrDCT of signal1", WINDOW_NORMAL);
	cvShowImage("Reconstruction from FrDCT of signal1", plot3.Figure);//重建信号的Plot

	waitKey(0);
	getchar();
	return;
} 

 /*  void main()
{
	char* file = "lena.bmp";
	int i, M, N;
	Mat picture = imread(file, IMREAD_GRAYSCALE);//强行转换为灰度图像
	if (!picture.data)
	{
		printf("\nFile %s read error\n", file);
		system("pause");
		return;
	}
	else
		printf("\nFile %s read successfully\n", file);

	//先转换下格式  
	Mat image, frDctImage, frDctImageRec, logFrDctImage;
	picture.convertTo(image, CV_64FC1);

	M = image.rows;
	N = image.cols;
	//密钥
	double u = 3.999, seed = 0.2;
	int ind = 1000;

	
	MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N / 2, 1, CV_64FC1);   //分数阶向量化
	MatND q = Mat::zeros(N / 2, 1, CV_64FC1);  //生成序列设为0，长度 N/2
	for (int i = 0; i < N / 2; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i + N / 2));
	}
	

	MatND t = chaos(seed, ind, u, 2*N);
	MatND a = Mat::zeros(N, 1, CV_64FC1);   //分数阶向量化
	MatND q = Mat::zeros(N, 1, CV_64FC1);  //生成序列设为0，长度 N/2
	for (int i = 0; i < N ; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i+N));
	}

	//设置猫映射的初始值Pd,Qd;
	//初始值
	 int K2 = 100;
	double u = 3.8899, seed = 0.2;
	int length = N*N;
	double Pd, Qd;
	//MatND v = Rerange(seed,K2,u,length);
	//Pd = v.at<double>(K2);
	cout << "qd=" << Pd << endl;
	Qd = v.at<double>(K2+1000);
	cout << "pd=" << Qd << endl;  

	

	
	MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N / 2, 1, CV_64FC1);   //分数阶向量化
	MatND q = Mat::zeros(N / 2, 1, CV_64FC1);  //生成序列设为0，长度 N/2
	for (int i = 0; i < N / 2; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i + N / 2));
	}
	

    //不同的混沌初始值所生成随机序列q
    //每行的q都不一样
	/*
	MatND DifRowQ = Mat::zeros(N, N/2, CV_64FC1);
	for (int k = 0; k < N; k++)
	{
		seed = seed + 0.0005;
		MatND t2 = chaos(seed, ind, u, N);
		MatND a2 = Mat::zeros(N / 2, 1, CV_64FC1);   //分数阶向量化
		MatND q2 = Mat::zeros(N / 2, 1, CV_64FC1);  //生成序列设为0，长度 N/2
		for (int i = 0; i < N / 2; i++)
		{
			a2.at<double>(i) = t2.at<double>(i);
			q2.at<double>(i) = round(t2.at<double>(i + N / 2));
		}
		for (int j = 0; j < N / 2; j++)
		{
			DifRowQ.at<double>(k, j) = q2.at<double>(j);
			//cout << "DifRowQ=" << DifRowQ.at<double>(k, j) << endl;
		}
	
	}
	
	//cout << "a=" << a << endl;

	//cout << "q=" << q << endl;

	
	frDctImage = frDct(image, a, q);
	frDctImageRec = frDct(frDctImage, -a, q);

	cout << "\nMSE= " << getMSE(frDctImageRec, image) << " dB";
	cout << "\nPSNR= " << getPSNR(frDctImageRec, image) << " dB";

	log(abs(frDctImage) + 1, logFrDctImage);
	normalize(logFrDctImage, logFrDctImage, 0, 255, CV_MINMAX, CV_8UC1);//归一化像素值[0,255] 
	normalize(frDctImageRec, frDctImageRec, 0, 255, CV_MINMAX, CV_8UC1);

	string windowOri = "Image original";
	string windowTrans = "Image FrDCT";
	string windowRec = "Image reconstruction";
	namedWindow(windowOri, WINDOW_NORMAL);
	imshow(windowOri, picture);
	namedWindow(windowTrans, WINDOW_NORMAL);
	imshow(windowTrans, logFrDctImage);
	namedWindow(windowRec, WINDOW_NORMAL);
	imshow(windowRec, frDctImageRec);

	Mat fimage;
	fimage = Quantification(frDctImage);
	string windowQua = "qualitifaction image";
	namedWindow(windowQua, WINDOW_NORMAL);
	imshow(windowQua, fimage);

	Mat difusionImage, logdifusionImage;
	difusionImage = Arnold(fimage, Pd, Qd);     //猫映射的初始值已知。
	normalize(difusionImage, logdifusionImage, 0, 255, CV_MINMAX, CV_8UC1);//归一化像素值[0,255] 

	normalize(difusionImage, logdifusionImage, 0, 255, CV_MINMAX, CV_8UC1);

	string windowArnold = "Image arnold";
	namedWindow(windowArnold, WINDOW_NORMAL);
	imshow(windowArnold, logdifusionImage);

	Mat iv_fimage;
	iv_fimage = Iv_Quantification(fimage,image);
	string windowIvQua = "Iv_qualitifaction image";
	namedWindow(windowIvQua, WINDOW_NORMAL);
	imshow(windowIvQua, iv_fimage);


	//waitKey(0);
	//getchar();
	//return;
	//下面开始调试FrDCT,上面的Lena图像显示和下面的程序没有什么关系

	N = 16; //信号长度
	MatND signal1 = Mat::zeros(N, 1, CV_64FC1);

	//通过取舍注释，选取信号
	for (i = 0; i < 3; i++)signal1.at<double>(i) = 1; //方波信号，3个1，13个0
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = cos(2 * pi*(i + 1) / 16);//余弦信号
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = (0.005*exp(i / 2.0) + 1)/11.0;//指数信号
	//cout << endl << "signal1 = " << signal1;
	//为了求分数DCT，先取得分数变换矩阵，这样在重复调用时，可以直接利用这个矩阵，节约时间
	Mat Ca;
	Ca = getFrDctCa(N, a, q);  //变换矩阵，分数阶a
	Mat Ca_;	
	Ca_ = getFrDctCa(N, -a, q);//逆变换矩阵，分数阶a

	Mat frDctSignal1 = Ca*signal1; //再做变换
	//cout << "\nFrDCT signal1 = \n" << frDctSignal1;//打印变换结果
	Mat frDctSignal1Rec = Ca_*frDctSignal1;//逆变换
	//cout << "\nsignal1 = \n" << signal1;//对比结果,验证分数阶为负数，即是逆变换
	//cout << "\nReconstruction from FrDCT of signal1\n" << frDctSignal1Rec;//打印变换结果
	double mse1 = getMSE(signal1, frDctSignal1Rec);//求均方误差


	Plot plot1;
	plot1.plot(signal1, CvScalar(255, 0, 0), 'd');
	namedWindow("Signal 1", WINDOW_NORMAL);
	cvShowImage("Signal 1", plot1.Figure);//原始信号plot

	Plot plot2;
	plot2.plot(frDctSignal1, CvScalar(0, 255, 0), 'd');
	namedWindow("FrDct of Signal 1", WINDOW_NORMAL);
	cvShowImage("FrDct of Signal 1", plot2.Figure);//原始信号的FrDCT的Plot

	Plot plot3;
	plot3.plot(frDctSignal1Rec, CvScalar(0, 0, 255), 'd');
	namedWindow("Reconstruction from FrDCT of signal1", WINDOW_NORMAL);
	cvShowImage("Reconstruction from FrDCT of signal1", plot3.Figure);//重建信号的Plot
	
	
	
	waitKey(0);
	
	getchar();
	return;
} */
