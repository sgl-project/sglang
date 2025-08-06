#pragma once

#include "cutlass/barrier.h"

namespace flashmla {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
    SReady = 1,
    SoftmaxReady = 2,
    TransVReady = 3,
};

} // flashmla
