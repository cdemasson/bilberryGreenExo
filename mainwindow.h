#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void computeGreenParse();
    void selectPicture();
    void computeGreenWithCuda();

private:
    Ui::MainWindow *ui;
    QImage *mPicToParse;
    QImage *mParsedPic;
};

#endif // MAINWINDOW_H
