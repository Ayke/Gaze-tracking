package com.example.arthur.android_gaze_tracking;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Point;
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
import org.opencv.core.Core;
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

  private static final Scalar FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
  private static final int cropWidth = 640, cropHeight = 360;

  private static final int[] RIGHT_EYE_LANDMARS_IDXS = {36, 42};

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

  void startCameraListener(){
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
        case LoaderCallbackInterface.SUCCESS:
        {
          Log.i(TAG, "OpenCV loaded successfully");
          mOpenCvCameraView.enableView();
        } break;
        default:
        {
          super.onManagerConnected(status);
        } break;
      }
    }
  };

  @Override
  public void onPause()
  {
    super.onPause();
    if (mOpenCvCameraView != null)
      mOpenCvCameraView.disableView();
  }

  @Override
  public void onResume()
  {
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

      Imgproc.rectangle(mRgba, new org.opencv.core.Point(tl[0], tl[1]), new org.opencv.core.Point(br[0], br[1]), FACE_RECT_COLOR, 3);

      // Get 68 landmark points
      ArrayList<Point> landmarks = ret.getFaceLandmarks();
      Log.w(TAG, "landmarks: " + landmarks.size());
      for (Point point : landmarks) {
        float[] fp = {point.x, point.y};
        cropToFrameTransform.mapPoints(fp);
        Imgproc.circle(mRgba, new org.opencv.core.Point(fp[0], fp[1]),2, FACE_RECT_COLOR);
      }

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

  private void detectGaze(Mat mRgba, ArrayList<Point> landmarks){
    org.opencv.core.Point[] points = new org.opencv.core.Point[RIGHT_EYE_LANDMARS_IDXS[1] - RIGHT_EYE_LANDMARS_IDXS[0]];
    for (int i=RIGHT_EYE_LANDMARS_IDXS[0]; i<RIGHT_EYE_LANDMARS_IDXS[1]; i++) {
      points[i] = new org.opencv.core.Point(landmarks.get(i).x, landmarks.get(i).y);
    }

    Rect rect = Imgproc.boundingRect(new MatOfPoint(points));
    Mat right_eye_frame = new Mat(mRgba, rect);
    Mat gray_right_eye_frame = new Mat();
    Imgproc.cvtColor(right_eye_frame, gray_right_eye_frame, Imgproc.COLOR_RGB2GRAY);

     Mat binary_right_eye_frame = new Mat();
     Imgproc.adaptiveThreshold(gray_right_eye_frame, binary_right_eye_frame,
             255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, th, thc);

    List<MatOfPoint> contours = new ArrayList<> ();
    Mat hierachy = new Mat();
    Imgproc.findContours(binary_right_eye_frame, contours, hierachy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);


    for (int i=0; i<contours.size(); i++) {
      MatOfPoint cnt = contours.get(i);
      double area = Imgproc.contourArea(cnt);
      if (area < 1e-5){
        contours.remove(i);
        break;
      }
    }

    int bestBlob = -1;
    if (contours.size() >= 2){
      double maxArea = -1;

      for (int i=0; i<contours.size(); i++) {
        MatOfPoint cnt = contours.get(i);
        double area = Imgproc.contourArea(cnt);
        if (area > maxArea){
          maxArea = area;
          bestBlob = i;
        }
      }
    }
    else if (contours.size() == 1)
      bestBlob = 0;
    else
      bestBlob = -1;

    if (bestBlob >= 0){
      Moments center = Imgproc.moments(contours.get(bestBlob));
      if (center["m00"] == 0)

            else:
      (cx, cy) = (int(center['m10']/center['m00']),
                            int(center['m01']/center['m00']))
      cv2.circle(right_eye_frame, (cx, cy), 3, (0, 255, 0), 1)
    }
  }

}
