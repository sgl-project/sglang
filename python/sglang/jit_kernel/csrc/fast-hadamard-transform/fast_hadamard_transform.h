/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
};
