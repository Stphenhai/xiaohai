#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "plot.h"//����plot.h��Ϊ�˻�ͼ
#include "frdct.h"//����frdct.h��Ϊ�˵��ñ�ʵ����DCT�����÷�����frdct.h

using namespace std;
using namespace cv;




void main()
{
	char* file = "lena.bmp";
	int i=0, M, N;
	Mat picture = imread(file, IMREAD_GRAYSCALE);//ǿ��ת��Ϊ�Ҷ�ͼ��
	if (!picture.data)
	{
		printf("\nFile %s read error\n", file);
		system("pause");
		return;
	}
	else
		printf("\nFile %s read successfully\n", file);

	//cout << picture.at<double>(1, 1) << endl;
	//��ת���¸�ʽ  
	Mat image, frDctImage,frDctImageRec,logFrDctImage;
	picture.convertTo(image, CV_64FC1);

	M = image.rows;
	N = image.cols;

	//��Կ
	double u = 3.999, seed = 0.2;
	int ind = 1000;
	
	/*MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N / 2, 1, CV_64FC1);   //������������
	MatND q = Mat::zeros(N / 2, 1, CV_64FC1);  //����������Ϊ0������ N/2
	for (int i = 0; i < N / 2; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i + N / 2));
	}*/

	//������������GS��һ�����
	//MatND t = chaos(seed, ind, u, 2 * N);
	//MatND a = Mat::zeros(N, 1, CV_64FC1);   //������������
	//MatND b= Mat::zeros(N, 1, CV_64FC1);
	//MatND q = Mat::zeros(N, 1, CV_64FC1);  //����������Ϊ0������ N/2
	//for (int i = 0; i < N; i++)
	//{
	//	a.at<double>(i) = t.at<double>(i);
	//	q.at<double>(i) = round(t.at<double>(i + N));
	//	b.at<double>(i) = 0.25;
	//}


	//������������GS�ڶ������
	MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N, 1, CV_64FC1);   //������������
	MatND b = Mat::zeros(N, 1, CV_64FC1);
	MatND q = Mat::zeros(N, 1, CV_64FC1);  //����������Ϊ0������ N/2
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
	////���ҷ���һ�����а���
	//double x0 = 0.2350,y0=0.3500,z0=0.7350,a0=0.0125,b0=0.0157,r=3.7700;
	//MatND F= threeDchaoticmap(x0,y0,z0,a0,b0,r);
	//int times = 70001;
	//MatND x = Mat::zeros(times, 1, CV_64FC1);   //������������
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

	//���ҷ�������èӳ��
	//double key1 = 225, qd = 153;
	MatND bix = chaos(seed, ind, u, N*N);
	for (int i = 0; i < N*N; i++) {
		bix.at<double>(i) = round(fmod(bix.at<double>(i) * 10000, 256));
	}
	cout << bix.at<double>(5)<<endl;
	Mat C= bix.reshape(N,N);
	cout << C.at<double>(120, 200)<<endl;
	

	frDctImage = frDct(image, a, q);
	//���³�������ͼ�������ֵ��������Сֵ
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(frDctImage, minp, maxp);
	cout << "Mat minv = " << minv << endl;
	cout << "Mat maxv = " << maxv << endl;
	
	//��������
	Mat lfrDctImage, ilfrDctImage;
	lfrDctImage = lianghua(frDctImage, maxv, minv);
	cout << "fr="<<frDctImage.at<double>(20, 50) << endl;
	
	Mat scrambleimage,rscrambleimage;
	Mat iscrambleimage;
	Mat finalencryptedimage, ifinalencryptedimage;
	scrambleimage = Arnold(lfrDctImage, C);    //èӳ��


	finalencryptedimage = imageXor(scrambleimage,C);  //ͼ��������
	
	//���ռ���ͼ�����ʾ
	Mat logFinalencryptedimage;
	logFinalencryptedimage= finalencryptedimage;
	//log(abs(finalencryptedimage) + 1, logFinalencryptedimage);    //ȡ��������������
	normalize(logFinalencryptedimage, logFinalencryptedimage, 0, 255, CV_MINMAX, CV_8UC1);//��һ������ֵ[0,255] 
	string windowFinal = "final encryptedimage";
	namedWindow(windowFinal, WINDOW_NORMAL);
	imshow(windowFinal, logFinalencryptedimage);

	ifinalencryptedimage = inverseimageXor(finalencryptedimage, C);     //��������
	//iscrambleimage = iArnold(scrambleimage, C);           //��ӳ��    �����������û�����࣬ͼ�����Ч���ǳ��ã�PSNR�ܴ�230.329
	iscrambleimage = iArnold(ifinalencryptedimage, C);

	//Mat scrambleimage;
	//scrambleimage = Arnold(frDctImage, C, pd, qd);
	////����2��ʾ���Һ��ͼ��
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
	normalize(scrambleimage, scrambleimage, 0, 255, CV_MINMAX, CV_8UC1);//��һ������ֵ[0,255] 
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


	//���濪ʼ����FrDCT,�����Lenaͼ����ʾ������ĳ���û��ʲô��ϵ

	N = 16; //�źų���
	MatND signal1 = Mat::zeros(N, 1, CV_64FC1);

	//ͨ��ȡ��ע�ͣ�ѡȡ�ź�
	for (i = 0;i < 3;i++)signal1.at<double>(i) = 1; //�����źţ�3��1��13��0
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = cos(2 * pi*(i + 1) / 16);//�����ź�
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = (0.005*exp(i / 2.0) + 1)/11.0;//ָ���ź�
	//cout << endl << "signal1 = " << signal1;
	//Ϊ�������DCT����ȡ�÷����任�����������ظ�����ʱ������ֱ������������󣬽�Լʱ��
	Mat Ca;
	Ca = getFrDctCa(N, a, q);  //�任���󣬷�����a
	Mat Cb;
	Cb = getFrDctCa(N, b, q);  //�任���󣬷�����b
	Mat Ca_;
	Ca_ = getFrDctCa(N, -a, q);//��任���󣬷�����a
	Mat Cb_;
	Cb_ = getFrDctCa(N, -b, q);//��任���󣬷�����b
	
	Mat frDctSignal1 = Ca*signal1; //�����任
	//cout << "\nFrDCT signal1 = \n" << frDctSignal1;//��ӡ�任���
	Mat frDctSignal1Rec = Ca_*frDctSignal1;//��任
	//cout << "\nsignal1 = \n" << signal1;//�ԱȽ��,��֤������Ϊ������������任
	//cout << "\nReconstruction from FrDCT of signal1\n" << frDctSignal1Rec;//��ӡ�任���
	double mse1 = getMSE(signal1, frDctSignal1Rec);//��������
	cout << endl << "Mse1 = " << mse1;//���mseԼ����0����任OK

	Mat Cab;
	Cab = getFrDctCa(N, a + b, q);//�任���󣬷�����a+b
	Mat frDctSignal2 = Cb*frDctSignal1;//Ca�Ľ������Cb
	Mat frDctSignal3 = Cab*signal1;//ֱ����Cab����֤�����׿ɼ���
	double mse2 = getMSE(frDctSignal2, frDctSignal3);//��������
	cout << endl << "Mse2 = " << mse2;//���mseԼ����0������ɼ���

	Mat C1;
	MatND c = Mat::zeros(N, 1, CV_64FC1);
	for (int i = 0; i < N; i++)
		c.at<double>(i) = i;
	C1 = getFrDctCa(N, c, q);
	Mat dctSignal;
	dct(signal1, dctSignal);//��ͨDCT
	double mse3 = getMSE(C1*signal1, dctSignal);//�ͷ�����Ϊ1ʱ��FrDCT�Ƚ�
	cout << endl << "Mse3 = " << mse3;//���mseԼ����0��֤��������Ϊ1ʱ�����ǳ���dct

	Plot plot1;
	plot1.plot(signal1, CvScalar(255, 0, 0), 'd');
	namedWindow("Signal 1", WINDOW_NORMAL);
	cvShowImage("Signal 1", plot1.Figure);//ԭʼ�ź�plot
	
	Plot plot2;
	plot2.plot(frDctSignal1, CvScalar(0, 255, 0), 'd');
	namedWindow("FrDct of Signal 1", WINDOW_NORMAL);
	cvShowImage("FrDct of Signal 1", plot2.Figure);//ԭʼ�źŵ�FrDCT��Plot

	Plot plot3;
	plot3.plot(frDctSignal1Rec, CvScalar(0, 0, 255), 'd');
	namedWindow("Reconstruction from FrDCT of signal1", WINDOW_NORMAL);
	cvShowImage("Reconstruction from FrDCT of signal1", plot3.Figure);//�ؽ��źŵ�Plot

	waitKey(0);
	getchar();
	return;
} 

 /*  void main()
{
	char* file = "lena.bmp";
	int i, M, N;
	Mat picture = imread(file, IMREAD_GRAYSCALE);//ǿ��ת��Ϊ�Ҷ�ͼ��
	if (!picture.data)
	{
		printf("\nFile %s read error\n", file);
		system("pause");
		return;
	}
	else
		printf("\nFile %s read successfully\n", file);

	//��ת���¸�ʽ  
	Mat image, frDctImage, frDctImageRec, logFrDctImage;
	picture.convertTo(image, CV_64FC1);

	M = image.rows;
	N = image.cols;
	//��Կ
	double u = 3.999, seed = 0.2;
	int ind = 1000;

	
	MatND t = chaos(seed, ind, u, N);
	MatND a = Mat::zeros(N / 2, 1, CV_64FC1);   //������������
	MatND q = Mat::zeros(N / 2, 1, CV_64FC1);  //����������Ϊ0������ N/2
	for (int i = 0; i < N / 2; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i + N / 2));
	}
	

	MatND t = chaos(seed, ind, u, 2*N);
	MatND a = Mat::zeros(N, 1, CV_64FC1);   //������������
	MatND q = Mat::zeros(N, 1, CV_64FC1);  //����������Ϊ0������ N/2
	for (int i = 0; i < N ; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i+N));
	}

	//����èӳ��ĳ�ʼֵPd,Qd;
	//��ʼֵ
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
	MatND a = Mat::zeros(N / 2, 1, CV_64FC1);   //������������
	MatND q = Mat::zeros(N / 2, 1, CV_64FC1);  //����������Ϊ0������ N/2
	for (int i = 0; i < N / 2; i++)
	{
		a.at<double>(i) = t.at<double>(i);
		q.at<double>(i) = round(t.at<double>(i + N / 2));
	}
	

    //��ͬ�Ļ����ʼֵ�������������q
    //ÿ�е�q����һ��
	/*
	MatND DifRowQ = Mat::zeros(N, N/2, CV_64FC1);
	for (int k = 0; k < N; k++)
	{
		seed = seed + 0.0005;
		MatND t2 = chaos(seed, ind, u, N);
		MatND a2 = Mat::zeros(N / 2, 1, CV_64FC1);   //������������
		MatND q2 = Mat::zeros(N / 2, 1, CV_64FC1);  //����������Ϊ0������ N/2
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
	normalize(logFrDctImage, logFrDctImage, 0, 255, CV_MINMAX, CV_8UC1);//��һ������ֵ[0,255] 
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
	difusionImage = Arnold(fimage, Pd, Qd);     //èӳ��ĳ�ʼֵ��֪��
	normalize(difusionImage, logdifusionImage, 0, 255, CV_MINMAX, CV_8UC1);//��һ������ֵ[0,255] 

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
	//���濪ʼ����FrDCT,�����Lenaͼ����ʾ������ĳ���û��ʲô��ϵ

	N = 16; //�źų���
	MatND signal1 = Mat::zeros(N, 1, CV_64FC1);

	//ͨ��ȡ��ע�ͣ�ѡȡ�ź�
	for (i = 0; i < 3; i++)signal1.at<double>(i) = 1; //�����źţ�3��1��13��0
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = cos(2 * pi*(i + 1) / 16);//�����ź�
	//for (i = 0;i < 16;i++)signal1.at<double>(i) = (0.005*exp(i / 2.0) + 1)/11.0;//ָ���ź�
	//cout << endl << "signal1 = " << signal1;
	//Ϊ�������DCT����ȡ�÷����任�����������ظ�����ʱ������ֱ������������󣬽�Լʱ��
	Mat Ca;
	Ca = getFrDctCa(N, a, q);  //�任���󣬷�����a
	Mat Ca_;	
	Ca_ = getFrDctCa(N, -a, q);//��任���󣬷�����a

	Mat frDctSignal1 = Ca*signal1; //�����任
	//cout << "\nFrDCT signal1 = \n" << frDctSignal1;//��ӡ�任���
	Mat frDctSignal1Rec = Ca_*frDctSignal1;//��任
	//cout << "\nsignal1 = \n" << signal1;//�ԱȽ��,��֤������Ϊ������������任
	//cout << "\nReconstruction from FrDCT of signal1\n" << frDctSignal1Rec;//��ӡ�任���
	double mse1 = getMSE(signal1, frDctSignal1Rec);//��������


	Plot plot1;
	plot1.plot(signal1, CvScalar(255, 0, 0), 'd');
	namedWindow("Signal 1", WINDOW_NORMAL);
	cvShowImage("Signal 1", plot1.Figure);//ԭʼ�ź�plot

	Plot plot2;
	plot2.plot(frDctSignal1, CvScalar(0, 255, 0), 'd');
	namedWindow("FrDct of Signal 1", WINDOW_NORMAL);
	cvShowImage("FrDct of Signal 1", plot2.Figure);//ԭʼ�źŵ�FrDCT��Plot

	Plot plot3;
	plot3.plot(frDctSignal1Rec, CvScalar(0, 0, 255), 'd');
	namedWindow("Reconstruction from FrDCT of signal1", WINDOW_NORMAL);
	cvShowImage("Reconstruction from FrDCT of signal1", plot3.Figure);//�ؽ��źŵ�Plot
	
	
	
	waitKey(0);
	
	getchar();
	return;
} */
