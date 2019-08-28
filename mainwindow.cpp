#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QPixmap>
#include <QSize>
#include <QDebug>
#include <QVector>
#include <QColor>
#include <QRgb>
#include <QPoint>
#include <QElapsedTimer>

extern "C"
double function(void);

extern "C"
int compareMatrices(int height, int width, int*r, int*g, int*b, int*green);

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    mPicToParse = new QImage(":/images/images/misae.png");
    mParsedPic = new QImage(mPicToParse->size(), mPicToParse->format());
    QPixmap pix(":/images/images/misae.png");
    ui->picToParse->setPixmap(pix);

    ui->misae->adjustSize();
    ui->cetelem->adjustSize();
    connect(ui->misae, SIGNAL(clicked()), this, SLOT(selectPicture()));
    connect(ui->cetelem, SIGNAL(clicked()), this, SLOT(selectPicture()));
    connect(ui->parseButton, SIGNAL(clicked()), this, SLOT(computeGreenParse()));
    connect(ui->cudaButton, SIGNAL(clicked()), this, SLOT(computeGreenWithCuda()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::computeGreenParse()
{
    QElapsedTimer timer;
    timer.start();
    const QSize picSize = mPicToParse->size();
    for(int i = 0; i < picSize.width(); i++)
    {
        for(int j = 0; j < picSize.height(); j++)
        {
            QColor currentColor = mPicToParse->pixelColor(i, j);

            if(currentColor.green() > 2 * currentColor.red() || currentColor.green() > 2 * currentColor.blue()) //The pixel is green
            {
                mParsedPic->setPixel(i, j, 0xFF000000); //The pixel becomes black
            }
            else //The pixel is not green
            {
                mParsedPic->setPixel(i, j, 0xFFFFFFFF); //The pixel becomes white
            }
        }
    }
    ui->timeValue->setText(QString::number(timer.elapsed()) + "ms");
    ui->resultPic->setPixmap(QPixmap::fromImage(*mParsedPic));
}

void MainWindow::selectPicture()
{
    QObject* mObj = sender();
    QPushButton* mButton = qobject_cast<QPushButton*>(mObj);
    QString picPath = ":/images/images/" + mButton->objectName() + ".png";
    delete(mPicToParse);
    delete(mParsedPic);
    mPicToParse = new QImage(picPath);
    mParsedPic = new QImage(mPicToParse->size(), mPicToParse->format());
    ui->picToParse->setPixmap(QPixmap(picPath));
}

void MainWindow::computeGreenWithCuda()
{
    const QSize picSize = mPicToParse->size();
    int w = picSize.width();
    int h = picSize.height();
    int *arrR = static_cast<int *>(malloc(h * w * sizeof(int *)));
    int *arrG = static_cast<int *>(malloc(h * w * sizeof(int *)));
    int *arrB = static_cast<int *>(malloc(h * w * sizeof(int *)));
    int *arrResult = static_cast<int *>(malloc(h * w * sizeof(int *)));

    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            arrR[i*w+j] = mPicToParse->pixelColor(j, i).red();
            arrG[i*w+j] = mPicToParse->pixelColor(j, i).green();
            arrB[i*w+j] = mPicToParse->pixelColor(j, i).blue();
        }
    }

    QElapsedTimer timer;
    timer.start();

    int result = 0;
    result = compareMatrices(h, w, arrR, arrG, arrB, arrResult);

    //then create the result QImage from arrResult
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            if(arrResult[i*w+j])
            {
                mParsedPic->setPixel(j, i, 0xFF000000);
            }
            else
            {
                mParsedPic->setPixel(j, i, 0xFFFFFFFF);
            }
        }
    }

    ui->timeValue->setText(QString::number(timer.elapsed()) + "ms");

    free(arrResult);
    ui->resultPic->setPixmap(QPixmap::fromImage(*mParsedPic));

}
