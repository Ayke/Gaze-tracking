package com.example.arthur.android_gaze_tracking;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.PointF;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import env.FileUtils;
import env.ImageUtils;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
  private static final String TAG = "MainActivity";

//  private static final String FACE_LANDMARKS_MODEL_FILE =
//          "file:///android_asset/shape_predictor_68_face_landmarks.dat";

  private static final int MY_PERMISSIONS_REQUEST_CAMERA = 10000;

  private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
  private static final int cropWidth = 640, cropHeight = 360;

  private static final int[] RIGHT_EYE_LANDMARS_IDXS = {36, 42};

  private static final int blocks = 2;

  private static final int runtime_buffer_size = 3;
  private static final int regulate_buffer_size = 3;
  private static final int eye_center_buffer_size = 3;
  private static final int thickness_breaking = 2;

  private PointF center_eye_pos, center;
  private PointF scalar = new PointF(-90,240); //A bit too much
//  private PointF scalar = new PointF(-45,120); //A bit too little
  Buffer runtime_buffer;
  Buffer regulate_buffer;
  Buffer eye_center_buffer;

  private FaceDet faceDet;


  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private CameraBridgeViewBase mOpenCvCameraView;
  private TextView mResultView;
  private boolean mIsJavaCamera = true;
  private MenuItem mItemSwitchCamera = null;

  Mat mRgba;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    Log.i(TAG, "called onCreate");
    super.onCreate(savedInstanceState);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    setContentView(R.layout.show_camera);

    mResultView = (TextView) findViewById(R.id.results);
    mOpenCvCameraView = (JavaCameraView) findViewById(R.id.show_camera_activity_java_surface_view);

    faceDet = new FaceDet(Constants.getFaceShapeModelPath());

    if (ContextCompat.checkSelfPermission(this,
            Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this,
              new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
              MY_PERMISSIONS_REQUEST_CAMERA);
    } else {
      startCameraListener();
    }

  }

  void startCameraListener() {
    mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

    mOpenCvCameraView.setCvCameraViewListener(this);
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
                                         String permissions[], int[] grantResults) {
    switch (requestCode) {
      case MY_PERMISSIONS_REQUEST_CAMERA: {
        if (grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
          startCameraListener();
        }
        return;
      }
    }
  }

  private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
    @Override
    public void onManagerConnected(int status) {
      switch (status) {
        case LoaderCallbackInterface.SUCCESS: {
          Log.i(TAG, "OpenCV loaded successfully");
          mOpenCvCameraView.enableView();
        }
        break;
        default: {
          super.onManagerConnected(status);
        }
        break;
      }
    }
  };

  @Override
  public void onPause() {
    super.onPause();
    if (mOpenCvCameraView != null)
      mOpenCvCameraView.disableView();
  }

  @Override
  public void onResume() {
    super.onResume();
    if (!OpenCVLoader.initDebug()) {
      Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
      OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
    } else {
      Log.d(TAG, "OpenCV library found inside package. Using it!");
      mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }
  }

  public void onDestroy() {
    super.onDestroy();
    if (mOpenCvCameraView != null)
      mOpenCvCameraView.disableView();
  }

  public void onCameraViewStarted(int width, int height) {
    if (!new File(Constants.getFaceShapeModelPath()).exists()) {
      Log.w(TAG, "Copying landmark model to " + Constants.getFaceShapeModelPath());
      FileUtils.copyFileFromRawToOthers(this, R.raw.shape_predictor_68_face_landmarks, Constants.getFaceShapeModelPath());
    }

    runtime_buffer = new Buffer(runtime_buffer_size);
    regulate_buffer = new Buffer(regulate_buffer_size);
    eye_center_buffer = new Buffer(eye_center_buffer_size);

//    Log.w(TAG, "width, height:" + width + " " + height); //1280 * 720

    center_eye_pos = new PointF(0,0);
    center = new PointF(width / 2, height / 2);

    mRgba = new Mat(height, width, CvType.CV_8UC4);

    rgbFrameBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    width, height,
                    cropWidth, cropHeight,
                    0, true);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
  }

  public void onCameraViewStopped() {
    mRgba.release();
  }

  public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
    mRgba = inputFrame.rgba();

    final long startTime = System.currentTimeMillis();

    rgbFrameBitmap = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);

    Utils.matToBitmap(mRgba, rgbFrameBitmap);
    final Canvas cropped_canvas = new Canvas(croppedBitmap);
    cropped_canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    List<VisionDetRet> results = faceDet.detect(croppedBitmap);
    for (final VisionDetRet ret : results) {
      String label = ret.getLabel();
      float[] tl = {ret.getLeft(), ret.getTop()};
      float[] br = {ret.getRight(), ret.getBottom()};
      cropToFrameTransform.mapPoints(tl);
      cropToFrameTransform.mapPoints(br);

//      Imgproc.rectangle(mRgba, new org.opencv.core.Point(tl[0], tl[1]), new org.opencv.core.Point(br[0], br[1]), FACE_RECT_COLOR, 3);

      // Get 68 landmark points
      ArrayList<Point> landmarks = ret.getFaceLandmarks();

      detectGaze(mRgba, landmarks);
    }


    final long endTime = System.currentTimeMillis();
    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        mResultView.setText("Time cost: " + String.valueOf((endTime - startTime) / 1000f) + " sec");
      }
    });

    return mRgba; // This function must return
  }

  private static int th = 35;
  private static double thc = 19;

  private Mat detectGaze(Mat mRgba, ArrayList<Point> crop_landmarks) {
    int right_eye_len = RIGHT_EYE_LANDMARS_IDXS[1] - RIGHT_EYE_LANDMARS_IDXS[0];
    org.opencv.core.Point[] right_eye = new org.opencv.core.Point[right_eye_len];
    for (int i = 0; i < right_eye_len; i++) {
      float[] fp = {crop_landmarks.get(RIGHT_EYE_LANDMARS_IDXS[0] + i).x, crop_landmarks.get(RIGHT_EYE_LANDMARS_IDXS[0] + i).y};
      cropToFrameTransform.mapPoints(fp);

      right_eye[i] = new org.opencv.core.Point(fp[0], fp[1]);
//      Imgproc.circle(mRgba, right_eye[i], 2, FACE_RECT_COLOR);
    }

    PointF rc = new PointF(0, 0);

    for (org.opencv.core.Point point : right_eye){
      rc.x += point.x; rc.y += point.y;
    }
    rc.x /= right_eye_len;
    rc.y /= right_eye_len;
    eye_center_buffer.add(rc);

    Rect rect = Imgproc.boundingRect(new MatOfPoint(right_eye));
    Mat right_eye_frame = new Mat(mRgba, rect);

    if (right_eye_frame.cols() == 0 || right_eye_frame.rows() == 0)
      return mRgba;

    Mat gray_right_eye_frame = new Mat();
    Imgproc.cvtColor(right_eye_frame, gray_right_eye_frame, Imgproc.COLOR_RGB2GRAY);

    Mat binary_right_eye_frame = new Mat();
    Imgproc.adaptiveThreshold(gray_right_eye_frame, binary_right_eye_frame,
            255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, th, thc);

    List<MatOfPoint> contours = new ArrayList<>();
    Mat hierachy = new Mat();
    Imgproc.findContours(binary_right_eye_frame, contours, hierachy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
      MatOfPoint cnt = contours.get(i);
      double area = Imgproc.contourArea(cnt);
      if (area < 1e-5) {
        contours.remove(i);
        break;
      }
    }

    int bestBlob = -1;
    if (contours.size() >= 2) {
      double maxArea = -1;

      for (int i = 0; i < contours.size(); i++) {
        MatOfPoint cnt = contours.get(i);
        double area = Imgproc.contourArea(cnt);
        if (area > maxArea) {
          maxArea = area;
          bestBlob = i;
        }
      }
    } else if (contours.size() == 1)
      bestBlob = 0;
    else
      bestBlob = -1;

    PointF c;

    if (bestBlob >= 0) {
      Moments center = Imgproc.moments(contours.get(bestBlob));
      if (center.m00 == 0)
        c = new PointF(0,0);
      else
        c = new PointF((float)(center.m10 / center.m00), (float)(center.m01 / center.m00));
      rc = eye_center_buffer.get();

      runtime_buffer.add(new PointF((float)(rect.x+c.x-rc.x), (float)(rect.y+c.y-rc.y)));

      Imgproc.circle(mRgba, new org.opencv.core.Point(rect.x+c.x, rect.y+c.y),5, new Scalar(255,0,0),4);
    } else {
      return mRgba;
    }

    return activate_pupil(mRgba, runtime_buffer.get());
  }


  private Mat activate_pupil(Mat mRgba, PointF point){
    int x = (int)((point.x - center_eye_pos.x)*scalar.x + center.x);
    int y = (int)((point.y - center_eye_pos.y)*scalar.y + center.y);
    if (x >= mRgba.cols())
      x = mRgba.cols()-3;
    if (x < 0)
      x = 0;
    if (y >= mRgba.rows())
      y = mRgba.rows()-3;
    if (y < 0)
      y = 0;

    Log.w(TAG, "x, y: "+x+" "+y + " point: " + point.x + " " + point.y);

    Imgproc.circle(mRgba, new org.opencv.core.Point(x, y),5, new Scalar(0,255,0),4);

    int col_break = mRgba.cols() / blocks, row_break = mRgba.rows() / blocks;

    int number = blocks*(y / row_break) + (x / col_break);

//    mRgba.setTo(new Scalar(0,0,0));
    for (int i = 0; i<blocks+1; i++)
      Imgproc.line(mRgba, new org.opencv.core.Point(i * col_break, 0), new org.opencv.core.Point(i * col_break, mRgba.rows()), new Scalar(255,255,255), thickness_breaking);

    for (int j = 0; j<blocks+1; j++)
      Imgproc.line(mRgba, new org.opencv.core.Point(0, j * row_break), new org.opencv.core.Point(mRgba.cols(), j * row_break), new Scalar(255,255,255), thickness_breaking);

    if (number < 0 || number >= blocks*blocks)
      return mRgba;

    int i = number % blocks;
    int j = number / blocks;
    Imgproc.rectangle(mRgba,
            new org.opencv.core.Point(col_break * i + thickness_breaking, row_break * j + thickness_breaking),
            new org.opencv.core.Point(col_break * (i+1) - thickness_breaking, row_break * (j+1) - thickness_breaking),
            new Scalar(255,0,0), 7);

    return mRgba;
  }

  class Buffer {
    int limit;
    int number = 0;
    PointF sum = new PointF(0,0);;
    int index = 0;
    ArrayList<PointF> q;

    Buffer(){this(5);}
    Buffer(int limit){
      this.limit = limit;
      q = new ArrayList<>(limit);
      for (int i = 0; i < limit; i++) {
        q.add(null);
      }
    }

    public void clear(){
      sum = new PointF(0,0);
      index = 0;
      q = new ArrayList<>();
      number = 0;
    }

    public void add(PointF point){
      sum.x += point.x;
      sum.y += point.y;

      if (q.get(index) != null){
        sum.x -= q.get(index).x;
        sum.y -= q.get(index).y;
      }
      q.set(index, point);
      index += 1;
      if (index == limit)
        index = 0;
      if (number < limit)
        number += 1;
    }

    public PointF get(){
      if (number == 0)
        return new PointF(0,0);
      else
        return new PointF(sum.x/number, sum.y/number);
    }
  }

}
