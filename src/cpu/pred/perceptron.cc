/*
 * Copyright (c) 2011, 2014 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Copyright (c) 2004-2006 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Kevin Lim
 *          Hannes Brantner
 */

#include "cpu/pred/perceptron.hh"
#include "base/bitfield.hh"
#include "base/intmath.hh"

PerceptronBP::PerceptronBP(const PerceptronBPParams *params) : BPredUnit(params) {
    globalHistory = std::vector<long long>(1, 0);
    globalHistoryBits = params->globalHistoryBits;
    historyRegisterMask = mask(globalHistoryBits);
    perceptronTableSize = params->predictorSize - globalHistoryBits;
    outputThreshold = floor(1.93 * globalHistoryBits + 14);
    bitsPerWeight = 1 + ceilLog2(outputThreshold);
    weightThreshold = (1 << ceilLog2(outputThreshold)) - 1;
    weightsPerPerceptron = globalHistoryBits + 1;
    perceptronSize = bitsPerWeight * weightsPerPerceptron;
    perceptronCount = 1 << floorLog2(perceptronTableSize / perceptronSize);
    perceptronTable = std::vector<std::vector<int>>(perceptronCount, std::vector<int>(weightsPerPerceptron, 0));
    perceptronTableMask = perceptronCount - 1;
}

inline
int
PerceptronBP::calculatePerceptronTableIndex(Addr &branch_addr, long long &globalHistory) {
    // Get low order bits after removing instruction offset.
    return ((branch_addr >> instShiftAmt) xor globalHistory) & perceptronTableMask;
}

inline
void
PerceptronBP::updateGlobalHistTaken(ThreadID tid) {
    globalHistory[tid] = (globalHistory[tid] << 1) | 1;
    globalHistory[tid] = globalHistory[tid] & historyRegisterMask;
}

inline
void
PerceptronBP::updateGlobalHistNotTaken(ThreadID tid) {
    globalHistory[tid] = (globalHistory[tid] << 1);
    globalHistory[tid] = globalHistory[tid] & historyRegisterMask;
}

std::vector<int>
PerceptronBP::computeInputVector(long long globalHistory) {
    std::vector<int> inputVector = {1};
    for (int i = globalHistoryBits - 1; i >= 0; i--) {
        if ((globalHistory >> i) & 1) {
            inputVector.push_back(1);
        } else {
            inputVector.push_back(-1);
        }
    }
    return inputVector;
}

int
PerceptronBP::getPerceptronOutput(std::vector<int> inputVector, std::vector<int> perceptronWeights) {
    return std::inner_product(std::begin(inputVector), std::end(inputVector), std::begin(perceptronWeights), 0);
}

inline
bool
PerceptronBP::getPrediction(int perceptronOutput) {
    return perceptronOutput >= 0;
}

void
PerceptronBP::btbUpdate(ThreadID tid, Addr branch_addr, void *&bp_history) {
    //Update Global History to Not Taken (clear LSB)
    globalHistory[tid] &= (historyRegisterMask & ~ULL(1));
}

bool
PerceptronBP::lookup(ThreadID tid, Addr branch_addr, void *&bp_history) {
    int perceptronTableIndex = calculatePerceptronTableIndex(branch_addr, globalHistory[tid]);
    std::vector<int> inputVector = computeInputVector(globalHistory[tid]);
    std::vector<int> perceptronWeights = perceptronTable[perceptronTableIndex];
    bool prediction = getPrediction(getPerceptronOutput(inputVector, perceptronWeights));

    // Create BPHistory and pass it back to be recorded.
    BPHistory *history = new BPHistory;
    history->prediction = prediction;
    history->globalHistory = globalHistory[tid];
    history->perceptronWeights = perceptronTable[perceptronTableIndex];
    bp_history = (void *) history;

    // Speculative update of the global history
    if (prediction) {
        updateGlobalHistTaken(tid);
        return true;
    } else {
        updateGlobalHistNotTaken(tid);
        return false;
    }
}

void
PerceptronBP::uncondBranch(ThreadID tid, Addr pc, void *&bp_history) {
    // Create BPHistory and pass it back to be recorded.
    BPHistory *history = new BPHistory;
    history->prediction = true;
    history->globalHistory = globalHistory[tid];
    history->perceptronWeights = perceptronTable[calculatePerceptronTableIndex(pc, globalHistory[tid])];
    bp_history = static_cast<void *>(history);

    updateGlobalHistTaken(tid);
}

void
PerceptronBP::update(ThreadID tid, Addr branch_addr, bool taken,
                     void *bp_history, bool squashed,
                     const StaticInstPtr &inst, Addr corrTarget) {
    assert(bp_history);

    BPHistory *history = static_cast<BPHistory *>(bp_history);

    // If this is a misprediction, restore the speculatively
    // updated state (global history register and local history)
    // and update again.
    if (squashed) {
        // Global history restore and update
        globalHistory[tid] = (history->globalHistory << 1) | taken;
        globalHistory[tid] &= historyRegisterMask;

        return;
    }

    // Update the perceptron weights
    int perceptronTableIndex = calculatePerceptronTableIndex(branch_addr, history->globalHistory);
    std::vector<int> perceptronWeights = history->perceptronWeights;
    std::vector<int> inputVector = computeInputVector(history->globalHistory);
    bool prediction = history->prediction;

    if (prediction != taken || std::abs(getPerceptronOutput(inputVector, perceptronWeights)) <= outputThreshold) {
        for (int i = 0; i < weightsPerPerceptron; i++) {
            if (taken) {
                perceptronTable[perceptronTableIndex][i] += inputVector[i];
            } else {
                perceptronTable[perceptronTableIndex][i] -= inputVector[i];
            }
            perceptronTable[perceptronTableIndex][i] =
                    std::min(perceptronTable[perceptronTableIndex][i], weightThreshold);
            perceptronTable[perceptronTableIndex][i] =
                    std::max(perceptronTable[perceptronTableIndex][i], -weightThreshold);
        }
    }

    // We're done with this history, now delete it.
    delete history;
}

void
PerceptronBP::squash(ThreadID tid, void *bp_history) {
    BPHistory *history = static_cast<BPHistory *>(bp_history);

    // Restore global history to state prior to this branch.
    globalHistory[tid] = history->globalHistory;

    // Delete this BPHistory now that we're done with it.
    delete history;
}

PerceptronBP *
PerceptronBPParams::create() {
    return new PerceptronBP(this);
}

#ifdef DEBUG
int
PerceptronBP::BPHistory::newCount = 0;
#endif
