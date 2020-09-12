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
 *          Timothy M. Jones
 *          Nilay Vaish
 *          Hannes Brantner
 */

#ifndef __CPU_PRED_PERCEPTRON_PRED_HH__
#define __CPU_PRED_PERCEPTRON_PRED_HH__

#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

#include "base/sat_counter.hh"
#include "base/types.hh"
#include "cpu/pred/bpred_unit.hh"
#include "params/PerceptronBP.hh"

/**
 * Implements a perceptron branch predictor, hopefully identical to the one
 * used in the 21264.  It has a local predictor, which uses a local history
 * table to index into a table of counters, and a global predictor, which
 * uses a global history to index into a table of counters.  A choice
 * predictor chooses between the two.  Both the global history register
 * and the selected local history are speculatively updated.
 */
class PerceptronBP : public BPredUnit {
public:
    /**
     * Default branch predictor constructor.
     */
    PerceptronBP(const PerceptronBPParams *params);

    /**
     * Looks up the given address in the branch predictor and returns
     * a true/false value as to whether it is taken.  Also creates a
     * BPHistory object to store any state it will need on squash/update.
     * @param branch_addr The address of the branch to look up.
     * @param bp_history Pointer that will be set to the BPHistory object.
     * @return Whether or not the branch is taken.
     */
    bool lookup(ThreadID tid, Addr branch_addr, void *&bp_history);

    /**
     * Records that there was an unconditional branch, and modifies
     * the bp history to point to an object that has the previous
     * global history stored in it.
     * @param bp_history Pointer that will be set to the BPHistory object.
     */
    void uncondBranch(ThreadID tid, Addr pc, void *&bp_history);

    /**
     * Updates the branch predictor with the actual result of a branch.
     * @param branch_addr The address of the branch to update.
     * @param taken Whether or not the branch was taken.
     * @param bp_history Pointer to the BPHistory object that was created
     * when the branch was predicted.
     * @param squashed is set when this function is called during a squash
     * operation.
     * @param inst Static instruction information
     * @param corrTarget Resolved target of the branch (only needed if
     * squashed)
     */
    void update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history,
                bool squashed, const StaticInstPtr &inst, Addr corrTarget);

    /**
     * Restores the global branch history on a squash.
     * @param bp_history Pointer to the BPHistory object that has the
     * previous global branch history in it.
     */
    void squash(ThreadID tid, void *bp_history);

    /**
     * Updates the branch predictor to Not Taken if a BTB entry is
     * invalid or not found.
     * @param branch_addr The address of the branch to look up.
     * @param bp_history Pointer to any bp history state.
     */
    void btbUpdate(ThreadID tid, Addr branch_addr, void *&bp_history);

private:
    /**
     * Returns if the branch should be taken or not, given a counter
     * value.
     * @param perceptronOutput The output of a single perceptron.
     */
    inline bool getPrediction(int perceptronOutput);

    /**
     * Returns the output of a single perceptron
     * @param inputVector The first vector for the dot product
     * @param perceptronWeights The second vector for the dot product
     */
    int getPerceptronOutput(std::vector<int> inputVector, std::vector<int> perceptronWeights);

    /**
     * Returns the input vector to compute the perceptron output
     * @param globalHistory The global history that is transformed.
     */
    std::vector<int> computeInputVector(long long globalHistory);

    /**
     * Returns the perceptron table index, given a branch address.
     * @param branch_addr The branch's PC address.
     */
    inline int calculatePerceptronTableIndex(Addr &branch_addr, long long &globalHistory);

    /** Updates global history as taken. */
    inline void updateGlobalHistTaken(ThreadID tid);

    /** Updates global history as not taken. */
    inline void updateGlobalHistNotTaken(ThreadID tid);

    /**
     * The branch history information that is created upon predicting
     * a branch.  It will be passed back upon updating and squashing,
     * when the BP can use this information to update/restore its
     * state properly.
     */
    struct BPHistory {
#ifdef DEBUG
        BPHistory()
        { newCount++; }
        ~BPHistory()
        { newCount--; }

        static int newCount;
#endif
        bool prediction;
        long long globalHistory;
        std::vector<int> perceptronWeights;
    };

    /** Number of perceptrons stored in the perceptron table. */
    int perceptronCount;

    /** Number of bits to store all the weights of a single perceptron. */
    int perceptronSize;

    /** Amount of weights per perceptron. */
    int weightsPerPerceptron;

    /** Threshold for the magnitude of a single perceptron weight. */
    int weightThreshold;

    /** Number of bits to store a single perceptron weight. */
    int bitsPerWeight;

    /** Threshold for the magnitude of the perceptron output. */
    int outputThreshold;

    /** Number of bits of the perceptron table. */
    int perceptronTableSize;

    /** Perceptron table. */
    std::vector <std::vector<int>> perceptronTable;

    /** Mask to index the perceptron table. */
    int perceptronTableMask;

    /** Global history register. Contains as much history as specified by
     *  globalHistoryBits. Actual number of bits used is determined by
     *  globalHistoryMask and choiceHistoryMask. */
    std::vector<long long> globalHistory;

    /** Number of bits for the global history. Determines maximum number of
        entries in global and choice predictor tables. */
    int globalHistoryBits;

    /** Mask to control how much history is stored. All of it might not be
     *  used. */
    long long historyRegisterMask;
};

#endif // __CPU_PRED_PERCEPTRON_PRED_HH__