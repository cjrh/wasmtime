// ISLE context helpers for `lower_ymm.isle` (fork-only), kept apart from the
// upstream helpers in `isle.rs`. `ymm_isle_helpers!` expands inside the
// `Context` impl.

macro_rules! ymm_isle_helpers {
    () => {
        fn put_in_xmm_mem_256(&mut self, val: Value) -> XmmMem256 {
            let ty = self.lower_ctx.dfg().value_type(val);
            assert_eq!(ty.bits(), 256, "put_in_xmm_mem_256: got value of type {ty}");
            // A sunk load reads exactly the 32 bytes the `load` would have.
            if let Some(load) = self.sinkable_load(val) {
                return XmmMem256::unwrap_new(RegMem::mem(self.sink_load(&load)));
            }
            XmmMem256::unwrap_new(RegMem::reg(self.put_in_reg(val)))
        }

        fn xmm_to_xmm_mem_256(&mut self, r: Xmm) -> XmmMem256 {
            XmmMem256::unwrap_new(RegMem::reg(r.to_reg()))
        }

        fn writable_xmm_to_xmm_mem_256(&mut self, r: WritableXmm) -> XmmMem256 {
            XmmMem256::unwrap_new(RegMem::reg(r.to_reg().to_reg()))
        }

        fn synthetic_amode_to_xmm_mem_256(&mut self, amode: &SyntheticAmode) -> XmmMem256 {
            XmmMem256::unwrap_new(RegMem::mem(amode.clone()))
        }

        fn const_to_xmm_mem_256(&mut self, c: VCodeConstant) -> XmmMem256 {
            XmmMem256::unwrap_new(RegMem::mem(SyntheticAmode::ConstantOffset(c)))
        }
    };
}
