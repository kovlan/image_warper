#include <Python.h>

#include <math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>


static PyObject* fastWarper_check(PyObject* self)//, PyObject* args)
{
    return Py_BuildValue("s", "Hello, Python extensions for fast image warping!!");
}

static void printArrayMetadata(PyArrayObject* arr)
{
    int nd = arr->nd;
    int dim1 = arr->dimensions[0];
    int dim2 = arr->dimensions[1];
    int dim3 = arr->dimensions[2];
    printf("nd = %d, dim1 = %d, dim2 = %d, dim3 = %d\n", nd, dim1, dim2, dim3);

    int stride1 = arr->strides[0];
    int stride2 = arr->strides[1];
    int stride3 = arr->strides[2];
    printf("nd = %d, stride1 = %d, stride2 = %d, stride3 = %d\n", nd, stride1, stride2, stride3);
}

static void warpVer(PyArrayObject* array, int width, int height)
{
    const double MAX_VALUE = 1000000.0;
    double* vCost[2];

    vCost[0] = malloc(width * sizeof(double));
    vCost[1] = malloc(width * sizeof(double));

    int vDirs[height * width];
    int curInd = 0, prevInd = 1;

    double* pa = (double*)array->data;

    int stride = array->strides[0] / 8;
    int pixelStride = array->strides[1] / 8;
    for (int y = 0; y < height; ++y) {
        vCost[curInd][0] = MAX_VALUE;
        vCost[curInd][width-1] = MAX_VALUE;

        double* pInLine = pa + y * stride;   
        int* pDirLine = vDirs + y * width;     
        for (int x = 1; x < width-1; ++x) {
            double value = 0.0;
            int xpx = x * pixelStride;
            for (int c = 0; c <= 2; ++c) {
                value += fabs(pInLine[xpx + c] - pInLine[xpx - pixelStride + c]);
                value += fabs(pInLine[xpx + c] - pInLine[xpx + pixelStride + c]);
            }
            vCost[curInd][x] = value;

            if (y > 0) {
                int argmin = x;
                if (vCost[prevInd][x-1] < vCost[prevInd][argmin]) {
                    argmin = x-1;
                }
                if (vCost[prevInd][x+1] < vCost[prevInd][argmin]) {
                    argmin = x+1;
                }

                vCost[curInd][x] += vCost[prevInd][argmin];
                pDirLine[x] = argmin;
            }
        }

        prevInd = 1 - prevInd;
        curInd = 1 - curInd;
    }

    int vErase[height];
    int bestPx = 0;
    double* p = vCost[prevInd];
    for (int x = 1; x < width-1; ++x) {
        if (p[x] < p[bestPx]) {
            bestPx = x;
        }
    }

    for (int y = height - 1; y >= 0; --y) {
        vErase[y] = bestPx;
        bestPx = vDirs[y * width + bestPx];
    }

    for (int y = 0; y < height; ++y) {
        double* pLine = pa + y * stride;   

        int erasePx = vErase[y];
        if (bestPx < width - 1) {
            memcpy(pLine + erasePx * pixelStride,
                   pLine + (erasePx + 1) * pixelStride,
                   (width - erasePx - 1) * array->strides[1]);
        }
    }

    free(vCost[0]);
    free(vCost[1]);
}

static void warpHor(PyArrayObject* array, int width, int height)
{
    const double MAX_VALUE = 1000000.0;
    double* vCost[2];

    vCost[0] = malloc(height * sizeof(double));
    vCost[1] = malloc(height * sizeof(double));

    int vDirs[height * width];
    int curInd = 0, prevInd = 1;

    double* pa = (double*)array->data;

    int stride = array->strides[0] / 8;
    int pixelStride = array->strides[1] / 8;
    for (int x = 0; x < width; ++x) {
        vCost[curInd][0] = MAX_VALUE;
        vCost[curInd][height-1] = MAX_VALUE;

        for (int y = 1; y < height-1; ++y) {
            double value = 0.0;
            double* p = pa + y * stride + x * pixelStride;
            for (int c = 0; c <= 2; ++c) {
                value += fabs(p[c] - p[c - pixelStride]);
                value += fabs(p[c] - p[c + pixelStride]);
            }
            vCost[curInd][y] = value;

            if (x > 0) {
                int argmin = y;
                if (vCost[prevInd][y-1] < vCost[prevInd][argmin]) {
                    argmin = y-1;
                }
                if (vCost[prevInd][y+1] < vCost[prevInd][argmin]) {
                    argmin = y+1;
                }

                vCost[curInd][y] += vCost[prevInd][argmin];
                vDirs[y * width + x] = argmin;
            }
        }

        prevInd = 1 - prevInd;
        curInd = 1 - curInd;
    }

    int vErase[width];
    int bestPx = 0;
    double* p = vCost[prevInd];
    for (int y = 1; y < height-1; ++y) {
        if (p[y] < p[bestPx]) {
            bestPx = y;
        }
    }

    for (int x = width - 1; x >= 0; --x) {
        vErase[x] = bestPx;
        bestPx = vDirs[bestPx * width + x];
    }

    for (int x = 0; x < width; ++x) {
        int erasePx = vErase[x];
        for (int y = erasePx; y < height - 1; ++y) {
            double* pDst = pa + y * stride + x * pixelStride;
            double* pSrc = pa + (y + 1) * stride + x * pixelStride;
            for (int c = 0; c <= 2; ++c) {
                pDst[c] = pSrc[c];
            }
        }
    }

    free(vCost[0]);
    free(vCost[1]);
}

static PyObject* fastWarper_warp(PyObject* self, PyObject* args)
{
    PyArrayObject *in_array = NULL;
    PyArrayObject *out_array = NULL;

    // parse input arguments, array, columns, rows
    int columns = -1, rows = -1;
    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &in_array, &columns, &rows))
    {
        return NULL;
    }

    //  construct the output array, like the input array
    out_array = (PyArrayObject*)PyArray_NewLikeArray(in_array, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL) {
        return NULL;
    }
    Py_INCREF(out_array);

    // copy in array to out array
    double* inP = (double*)in_array->data;
    double* outP = (double*)out_array->data;
    for (int y = 0; y < in_array->dimensions[0]; ++y) {
        double* pInLine = inP + y * in_array->strides[0] / 8;
        double* pOutLine = outP + y * out_array->strides[0] / 8;
        memcpy(pOutLine, pInLine, in_array->strides[0]);
    }

    int h = out_array->dimensions[0];
    int w = out_array->dimensions[1];
    int rc = rows < columns ? rows : columns;

    for (int i = 0; i < rc; ++i) {
        warpVer(out_array, w - i, h - i);        
        warpHor(out_array, w - i - 1, h - i);        
    }
        
    for (int c = 0; c < columns - rc; ++c) {
        warpVer(out_array, w - rc - c, h - rc);        
    }

    for (int r = 0; r < rows - rc; ++r) {
        warpHor(out_array, w - columns, h - rc - r);        
    }

    return (PyObject*)out_array;
}


static PyMethodDef fastWarper_methods[] = {
    { "check", (PyCFunction)fastWarper_check, METH_NOARGS, NULL },
    { "warp", fastWarper_warp, METH_VARARGS, "evaluate the cosine on a numpy array"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fastWarper",
    "This is a module 55",
    -1,                  
    fastWarper_methods   
};

PyMODINIT_FUNC PyInit_fastWarper() {
    Py_Initialize();
    import_array();
    return PyModule_Create(&moduledef);
}