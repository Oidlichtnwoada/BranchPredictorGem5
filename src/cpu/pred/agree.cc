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

#include "cpu/pred/agree.hh"
#include "base/bitfield.hh"
#include "base/intmath.hh"

AgreeBP::AgreeBP(const AgreeBPParams *params) : BPredUnit(params) {
    biasingBitsTable = {};
    localCounterBits = params->localCounterBits;
    counterAmount = 1 << floorLog2((params->predictorSize - params->BTBEntries) / localCounterBits);
    counterTable = std::vector<SatCounter>(counterAmount, SatCounter(localCounterBits));
    counterTableMask = counterAmount - 1;
    globalHistory = std::vector<long long>(1, 0);
    globalHistoryBits = log2(counterAmount);
    historyRegisterMask = mask(globalHistoryBits);
}

inline
int
AgreeBP::calculateCounterTableIndex(Addr &branch_addr, long long &globalHistory) {
    // Get low order bits after removing instruction offset.
    return ((branch_addr >> instShiftAmt) xor globalHistory) & counterTableMask;
}

inline
void
AgreeBP::updateGlobalHistTaken(ThreadID tid) {
    globalHistory[tid] = (globalHistory[tid] << 1) | 1;
    globalHistory[tid] = globalHistory[tid] & historyRegisterMask;
}

inline
void
AgreeBP::updateGlobalHistNotTaken(ThreadID tid) {
    globalHistory[tid] = (globalHistory[tid] << 1);
    globalHistory[tid] = globalHistory[tid] & historyRegisterMask;
}

void
AgreeBP::btbUpdate(ThreadID tid, Addr branch_addr, void *&bp_history) {
    //Update Global History to Not Taken (clear LSB)
    globalHistory[tid] &= (historyRegisterMask & ~ULL(1));

    //Remove biasing bit from biasingBitsTable if branch_addr is not in BTB
    biasingBitsTable.erase(branch_addr);
}

bool AgreeBP::getAgreement(int counter) {
    return (counter >> (localCounterBits - 1)) & 1;
}

bool AgreeBP::getBias(Addr branch_addr) {
    if (biasingBitsTable.find(branch_addr) != biasingBitsTable.end()) {
        return biasingBitsTable.at(branch_addr);
    } else {
        return branch_addr > BTB.lookup(branch_addr, 0).pc();
    }
}

bool
AgreeBP::lookup(ThreadID tid, Addr branch_addr, void *&bp_history) {
    int counterTableIndex = calculateCounterTableIndex(branch_addr, globalHistory[tid]);
    bool agreement = getAgreement(counterTable[counterTableIndex]);
    bool bias = getBias(branch_addr);
    bool prediction = agreement == bias;

    // Create BPHistory and pass it back to be recorded.
    BPHistory *history = new BPHistory;
    history->globalHistory = globalHistory[tid];
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
AgreeBP::uncondBranch(ThreadID tid, Addr pc, void *&bp_history) {
    // Create BPHistory and pass it back to be recorded.
    BPHistory *history = new BPHistory;
    history->globalHistory = globalHistory[tid];
    bp_history = static_cast<void *>(history);

    updateGlobalHistTaken(tid);
}

void
AgreeBP::update(ThreadID tid, Addr branch_addr, bool taken,
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

    //Set biasing bit in table
    if (biasingBitsTable.find(branch_addr) == biasingBitsTable.end()) {
        biasingBitsTable.emplace(branch_addr, taken);
    }

    //Update the counterTable with the acquired knowledge
    int counterTableIndex = calculateCounterTableIndex(branch_addr, history->globalHistory);
    if (getBias(branch_addr) == taken) {
        counterTable[counterTableIndex]++;
    } else {
        counterTable[counterTableIndex]--;
    }

    // We're done with this history, now delete it.
    delete history;
}

void
AgreeBP::squash(ThreadID tid, void *bp_history) {
    BPHistory *history = static_cast<BPHistory *>(bp_history);

    // Restore global history to state prior to this branch.
    globalHistory[tid] = history->globalHistory;

    // Delete this BPHistory now that we're done with it.
    delete history;
}

AgreeBP *
AgreeBPParams::create() {
    return new AgreeBP(this);
}

#ifdef DEBUG
int
AgreeBP::BPHistory::newCount = 0;
#endif
