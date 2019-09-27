//Usage:
//Plot plot;
//plot.plot(signal, color, c);
//cvShowImage(“imageWindow",plot.Figure);
//where color is color, e.g. calar（0，0，255),red,（BGR format）。
//	cline type '.', 'o', 'l', ......
//l          line, 直线,连续曲线	
//*          asterisk, 星 
//.          dot,点 
//o          Circle,圈 
//x          x,叉 
//+          cross, 十字 
//s          square，方块 
//r          rhombus,菱形
//d          discrete, digital, 离散信号
//h          histogram,直方图

#ifndef _PLOT_H
#define _PLOT_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class Plot
{
	private:	
	void DrawAxis (IplImage *image);
	void DrawData (IplImage *image);
	int window_height;
	int window_width;
	vector< vector<CvPoint2D64f> >dataset;	
	vector<char> lineTypeSet;
	//color
	CvScalar backgroud_color;
	CvScalar axis_color;
	CvScalar text_color;
	public:
	IplImage* Figure;
	// manual or automatic range
	bool custom_range_y;
	double y_max;
	double y_min;
	double y_scale;
	bool custom_range_x;
	double x_max;
	double x_min;
	double x_scale;
	//边界大小
	int border_size;
	template<class T>
	void plot(IplImage* image, T *y, size_t Cnt, char lineType='l');	
	template<class T>
	void plot(T *x, T *y, size_t Cnt, char lineType='l');
	void xlabel(string xlabel_name, CvScalar label_color);
	void ylabel(string ylabel_name, CvScalar label_color);
	void plot(MatND signal,CvScalar color,char lineType);
	//清空图片上的数据
	void clear();
	void title(string title_name, CvScalar title_color);
	Plot();
	~Plot();
};
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//采用范型设计，因此将实现部分和声明部分放在一个文件中
Plot::Plot()
{
	this->border_size = 30;
	this->window_height = 600;
	this->window_width = 600;;
	this->Figure = cvCreateImage(cvSize(this->window_height, this->window_width),IPL_DEPTH_8U, 3);
	memset(Figure->imageData, 255, sizeof(unsigned char)*Figure->widthStep*Figure->height);
	//color
	this->backgroud_color = CV_RGB(255,255,255);
	this->axis_color = CV_RGB(0,0,0);
}

Plot::~Plot()
{
}
//范型设计
template<class T>
void Plot::plot(T *X, T *Y, size_t Cnt, char lineType)
{
	//对数据进行存储
	T tempX, tempY;
	vector<CvPoint2D64f>data;
	for(int i = 0; i < Cnt;i++)
	{
	tempX = X[i];
	tempY = Y[i];
	data.push_back(cvPoint2D64f((double)tempX, (double)tempY));
	}
	this->dataset.push_back(data);
	this->lineTypeSet.push_back(lineType);
	//printf("data count:%d\n", this->dataset.size());
	this->DrawData(this->Figure);
}
//TODO
void Plot::plot(MatND signal,CvScalar color, char lineType)
{
	size_t Cnt;
	double* X = new double[signal.rows * sizeof(double)];
	double* Y = new double[signal.rows * sizeof(double)];
	Cnt = signal.rows;
	signal.convertTo(signal, CV_64F, 1, 0);
	for (int i = 0;i < signal.rows;i++)
	{
		X[i] = (double)(i);
		Y[i] = signal.at<double>(i);
	}
	this->text_color = color;
	plot(X, Y, Cnt,lineType);
}

template<class T>
void Plot::plot(IplImage* image, T *Y, size_t Cnt, char lineType)
{
	//对数据进行存储
	T tempX, tempY;
	vector<CvPoint2D64f>data;
	for(int i = 0; i < Cnt;i++)
	{
		tempX = i;
		tempY = Y[i];
		data.push_back(cvPoint2D64f((double)tempX, (double)tempY));
	}
	this->dataset.push_back(data);
	this->lineTypeSet.push_back(lineType);
	//printf("data count:%d\n", this->dataset.size());
	this->DrawData(this->Figure);
}
void Plot::clear()
{
	this->dataset.clear();
	//memset(Figure->imageData, 255, sizeof(unsigned char)*Figure->widthStep*Figure->height);
}
void Plot::DrawAxis (IplImage *image)
{
	CvScalar axis_color = this->axis_color;
	CvScalar text_color = this->axis_color;
	int bs = this->border_size;		
	int h = this->window_height;
	int w = this->window_width;
	// size of graph
	int gh = h - bs * 2;
	int gw = w - bs * 2;
	int x, y, y1;
	// draw the horizontal and vertical axis
	// let x, y axies cross at zero if possible.
	double y_ref = this->y_min;
	if ((this->y_max > 0) && (this->y_min <= 0))
	y_ref = 0;
	int x_axis_pos = h - bs - cvRound((y_ref - this->y_min) * this->y_scale);
	//draw x axis
	cvLine(image, cvPoint(bs - 4, x_axis_pos), cvPoint(w - bs + 2, x_axis_pos), axis_color);
	//draw y axis
	cvLine(image, cvPoint(bs, h - bs), cvPoint(bs, h - bs - gh - 2), axis_color);
	//draw scales in x axis
	y = (int)(x_axis_pos + 1);
	for (int i = 0;i < this->x_max;i++)
	{
		x = (int)(bs + i*this->x_scale);
		if (i % 100 == 0)y1 = y + 5;
		else if (i % 50 == 0)y1 = y + 3;
			else  if (i % 10 == 0)y1 = y + 1;
				else continue;
		cvLine(image, cvPoint(x, y),cvPoint(x, y1), axis_color);
	}
	//
	// Draw the scale of the y axis
		CvFont font;
		cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,0.7, 0,1,CV_AA);

		int chw = 6, chh = 10;
		char text[16];

		// y max
		if ((this->y_max - y_ref) > 0.05 * (this->y_max - this->y_min))
		{
			snprintf(text, sizeof(text)-1, "%.1f", this->y_max);
			cvPutText(image, text, cvPoint(bs / 5, bs - chh / 2), &font, text_color);
		}
		// y min
		if ((y_ref - this->y_min) > 0.05 * (this->y_max - this->y_min))
		{
			snprintf(text, sizeof(text)-1, "%.1f", this->y_min);
			cvPutText(image, text, cvPoint(bs / 5, h - bs + chh), &font, text_color);
		}

		// x axis
		snprintf(text, sizeof(text)-1, "%.1f", y_ref);
		cvPutText(image, text, cvPoint(bs / 5, x_axis_pos + chh / 2), &font, text_color);

		// Write the scale of the x axis
		snprintf(text, sizeof(text)-1, "%.0f", this->x_max );
		cvPutText(image, text, cvPoint((int)(w - bs - strlen(text) * chw), (int)(x_axis_pos + chh)), 
				  &font, text_color);

		// x min
		snprintf(text, sizeof(text)-1, "%.0f", this->x_min );
		cvPutText(image, text, cvPoint(bs, x_axis_pos + chh), 
				  &font, text_color);
}

//添加对线型的支持
//TODO线型未补充完整
//标记		线型
//l          line, 直线,连续曲线	
//*          asterisk, 星 
//.          dot,点 
//o          Circle,圈 
//x          x,叉 
//+          cross, 十字 
//s          square，方块 
//r          rhombus,菱形
//d          discrete, digital, 离散信号
//h          histogram,直方图
void Plot::DrawData (IplImage *image)
{
	this->x_min = this->x_max = this->dataset[0][0].x;
	this->y_min = this->y_max = this->dataset[0][0].y;
	
	int bs = this->border_size;
	for(size_t i = 0; i < this->dataset.size(); i++)
	{
		for(size_t j = 0; j < this->dataset[i].size(); j++)
		{
			if(this->dataset[i][j].x < this->x_min)
			{
				this->x_min = this->dataset[i][j].x;
			}else if(this->dataset[i][j].x > this->x_max)
			{
				this->x_max = this->dataset[i][j].x;
			}
		
			if(this->dataset[i][j].y < this->y_min)
			{
				this->y_min = this->dataset[i][j].y;
			}else if(this->dataset[i][j].y > this->y_max)
			{
				this->y_max = this->dataset[i][j].y;
			}
		}
	}
	double x_range = this->x_max - this->x_min;
	double y_range = this->y_max - this->y_min;
	this->x_scale = (image->width - bs*2)/x_range;
	this->y_scale = (image->height- bs*2)/y_range;
	
	
	//清屏
	//memset(image->imageData, 255, sizeof(unsigned char)*Figure->widthStep*Figure->height);
	this->DrawAxis(image);
	
	//printf("x_range: %f y_range: %f\n", x_range, y_range);
	//绘制点
	double tempX, tempY, next_tempY;
	CvPoint prev_point, current_point, next_point;
	int radius = 3;
	int slope_radius = (int)(radius*1.414/2 + 0.5);
	for(size_t i = 0; i < (this->dataset.size()); i++)
	{
		//printf("dataset[i].size(): %d\n", dataset[i].size());	
		for(size_t j = 0; j < this->dataset[i].size(); j++)
		{
			tempX = (int)((this->dataset[i][j].x - this->x_min)*this->x_scale);
			tempY = (int)((this->dataset[i][j].y - this->y_min)*this->y_scale);
			next_tempY = (int)((0.0 - this->y_min)*this->y_scale);
			current_point = cvPoint((int)(bs + tempX), (int)(image->height - (tempY + bs)));
			next_point = cvPoint((int)(bs + tempX), (int)(image->height - (next_tempY + bs)));
			if(this->lineTypeSet[i] == 'l')//line
			{
				// draw a line between two points
				if (j >= 1)
				{
					cvLine(image, prev_point, current_point, this->text_color, 1, CV_AA);
				}		
				prev_point = current_point;
			}else if(this->lineTypeSet[i] == '.')
			{
				cvCircle(image, current_point, 1, this->text_color, -1, 8);
			}else if(this->lineTypeSet[i] == '*')
			{
				cvLine(image, cvPoint(current_point.x - slope_radius, current_point.y - slope_radius),
					cvPoint(current_point.x + slope_radius, current_point.y + slope_radius), this->text_color, 1, 8);
				cvLine(image, cvPoint(current_point.x - slope_radius, current_point.y + slope_radius),
					cvPoint(current_point.x + slope_radius, current_point.y - slope_radius), this->text_color, 1, 8);
				cvLine(image, cvPoint(current_point.x - radius + 1, current_point.y),
					cvPoint(current_point.x + radius - 1, current_point.y), this->text_color, 1, 8);
				cvLine(image, cvPoint(current_point.x, current_point.y - radius - 1),
					cvPoint(current_point.x, current_point.y + radius + 1), this->text_color, 1, 8);

			}else if(this->lineTypeSet[i] == 'o')//circl
			{
				cvCircle(image, current_point, radius, this->text_color, 1, CV_AA);
			}else if(this->lineTypeSet[i] == 'x')//cross
			{
				cvLine(image, cvPoint(current_point.x - slope_radius, current_point.y - slope_radius), 
					   cvPoint(current_point.x + slope_radius, current_point.y + slope_radius), this->text_color, 1, 8);
					   
				cvLine(image, cvPoint(current_point.x - slope_radius, current_point.y + slope_radius), 
					   cvPoint(current_point.x + slope_radius, current_point.y - slope_radius), this->text_color, 1, 8);
			}else if(this->lineTypeSet[i] == '+')
			{
				cvLine(image, cvPoint(current_point.x - radius, current_point.y), 
					   cvPoint(current_point.x + radius, current_point.y), this->text_color, 1, 8);			   
				cvLine(image, cvPoint(current_point.x, current_point.y - radius), 
					   cvPoint(current_point.x, current_point.y + radius), this->text_color, 1, 8);
			}else if(this->lineTypeSet[i] == 's')//square
			{
				cvRectangle(image, cvPoint(current_point.x - slope_radius, current_point.y - slope_radius), 
					   cvPoint(current_point.x + slope_radius, current_point.y + slope_radius), this->text_color, 1, 8);
			}else if(this->lineTypeSet[i] == 'r')//rhombus
			{
				cvLine(image, cvPoint(current_point.x - radius, current_point.y),
					cvPoint(current_point.x, current_point.y - radius), this->text_color, 1, 8);
				cvLine(image, cvPoint(current_point.x - radius, current_point.y),
					cvPoint(current_point.x, current_point.y + radius), this->text_color, 1, 8);
				cvLine(image, cvPoint(current_point.x + radius, current_point.y),
					cvPoint(current_point.x, current_point.y - radius), this->text_color, 1, 8);
				cvLine(image, cvPoint(current_point.x + radius, current_point.y),
					cvPoint(current_point.x, current_point.y + radius), this->text_color, 1, 8);
			}
			else if (this->lineTypeSet[i] == 'h')//histogram
			{
				int barWidth = (window_width - bs) / (int)this->dataset[i].size() / 2;
				cvRectangle(image, cvPoint(current_point.x - barWidth, current_point.y - barWidth),
					cvPoint(current_point.x + barWidth, window_height-this->border_size-1), this->text_color, -1, 8);
			}
			else if (this->lineTypeSet[i] == 'd')//discrete signal
			{	
				cvLine(image, current_point, next_point, this->text_color, 1, 4);
				cvCircle(image, current_point, radius, this->text_color, 1, CV_AA);
			}
		}
	}	
}

void Plot::title(string title_name, CvScalar title_color = Scalar(0, 0, 0))
{
	int chw = 6, chh = 10;
	IplImage *image = this->Figure;
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 2, 0.7, 0, 1, CV_AA);
	int x = (int)((this->window_width - 2 * this->border_size) / 2 + this->border_size - (title_name.size() / 2.0) * chw);
	int y = (int)(this->border_size / 2 + 1);
	cvPutText(image, title_name.c_str(), cvPoint(x, y), &font, title_color);
}

void Plot::xlabel(string xlabel_name, CvScalar label_color = Scalar(0, 0, 0))
{
	int chw = 6, chh = 10;
	int bs = this->border_size;
	int h = this->window_height;
	int w = this->window_width;
	// let x, y axies cross at zero if possible.
	double y_ref = this->y_min;
	if ((this->y_max > 0) && (this->y_min <= 0))
	{
		y_ref = 0;
	}
	int x_axis_pos = h - bs - cvRound((y_ref - this->y_min) * this->y_scale);
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.5, 0.7, 0, 1, CV_AA);
	int x = (int)(this->window_width - this->border_size - chw * xlabel_name.size());
	int y = (int)(x_axis_pos + bs / 1.5);
	cvPutText(this->Figure, xlabel_name.c_str(), cvPoint(x, y+3), &font, label_color);
}
void Plot::ylabel(string ylabel_name, CvScalar label_color = Scalar(0, 0, 0))
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.5, 0.7, 0, 1, CV_AA);
	int x = this->border_size;
	int y = this->border_size;
	cvPutText(this->Figure, ylabel_name.c_str(), cvPoint(x, y-3), &font, label_color);
}

#endif