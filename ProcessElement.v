`timescale 1ns / 1ps

module processing_element #(
    parameter D_WIDTH = 16,
    parameter ACC_WIDTH = 32,
    parameter MAC_LANES = 8,
    parameter ADDR_WIDTH = 9,
    parameter FRAC_BITS = 12
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // GLOBAL FSM CONTROL SIGNALS (SIMD Bus)
    input  wire                 ce_en,
    input  wire                 ce_clear_acc,
    input  wire                 ldm_we,
    input  wire                 fsm_op_mode,
    input  wire                 fsm_relu_en,
    input  wire [ADDR_WIDTH-1:0] ldm_rd_addr,
    input  wire [ADDR_WIDTH-1:0] ldm_wr_addr,
    input  wire [2:0]            ldm_wr_lane_sel,

    // HALO RING CONTROL & DATA PORTS
    input  wire                 simd_halo_latch,   // FSM tells PE to grab neighbor's data
    input  wire                 simd_halo_mux_sel, // 0 = Read LDM, 1 = Read Halo Reg
    input  wire [127:0]         halo_in,           // Data arriving from Left Neighbor
    output wire [127:0]         halo_out,          // Data leaving to Right Neighbor
    
    // GLOBAL WEIGHT BROADCAST BUS
    input  wire signed [D_WIDTH-1:0] w0, w1, w2, w3, w4, w5, w6, w7
);

    // Internal routing wires between the LDM , the HALO and the CE
    wire [127:0]         ldm_to_ce;
    wire signed [31:0]   ce_raw_acc;
    wire signed [15:0]   ce_acc;
    wire signed [15:0]   ce_to_ldm;
    reg [127:0] halo_reg;
    // DATAPATH BRIDGE: ReLU, Pooling, & Muxing
    wire signed [15:0] conv_relu_result;
    wire signed [15:0] pool_result;

    // DATAPATH BRIDGE: Saturation & Truncation
    // The Light-CE accumulates into a 32-bit register to prevent overflow.
    // Before writing back to the 16-bit LDM, we must truncate or saturate.
    // (For this blueprint, we are simply slicing the lower 16 bits, but for 
    // production tape-out, you would insert fixed-point scaling/saturation here).
    assign halo_out = ldm_to_ce;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)      halo_reg <= 128'd0;
        else if (simd_halo_latch) halo_reg <= halo_in;
    end
    
    // 3. The Datapath Mux: Choose between our own memory or the neighbor's pixel
    wire [127:0] active_compute_data = simd_halo_mux_sel ? halo_reg : ldm_to_ce;

    // Multiplexer that chooses between which result to take
    assign ce_to_ldm = (fsm_op_mode == 1'b1) ? pool_result : conv_relu_result;

        // 1. Hardware ReLU & Truncation on the MAC output
    // If relu_en is high AND the sign bit (bit 31) is 1 (negative), force to 0.
    assign ce_acc = ce_raw_acc >>> FRAC_BITS;
    assign conv_relu_result = (fsm_relu_en && ce_raw_acc[31]) ? 16'd0 : ce_acc;

    // 2. Instantiate the Vector Max Pool Engine
    spatial_max_pool_8way #(
        .D_WIDTH(D_WIDTH),
        .LANES(MAC_LANES)
    ) u_max_pool (
        .clk(clk),
        .rst_n(rst_n),
        .pool_en(fsm_op_mode == 1'b1), // Enable only if FSM is in Pool Mode
        .pool_clear(ce_clear_acc),     // Re-use the existing clear signal
        .lane_sel(ldm_wr_lane_sel),
        .ldm_read_data(active_compute_data),     // Taps into the same 128-bit bus as the CE
        .pool_out(pool_result)
    );

    // INSTANTIATE: Unified Scratchpad LDM
    unified_scratchpad_ldm #(
        .D_WIDTH(D_WIDTH),
        .MAC_LANES(MAC_LANES),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_ldm (
        .clk(clk),
        .we(ldm_we),
        .rd_addr(ldm_rd_addr),
        .wr_addr(ldm_wr_addr),
        .data_in(ce_to_ldm),
        .wr_lane_sel(ldm_wr_lane_sel),
        .data_out_packed(ldm_to_ce)
    );

    // INSTANTIATE: 8-MAC Light Convolution Engine
    light_ce_8mac #(
        .D_WIDTH(D_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_light_ce (
        .clk(clk),
        .rst_n(rst_n),
        .en(ce_en),
        .clear_acc(ce_clear_acc),
        
        // Route the 8 global weights
        .w0(w0), .w1(w1), .w2(w2), .w3(w3), .w4(w4), .w5(w5), .w6(w6), .w7(w7),
        
        // Unpack the 128-bit LDM bus into the 8 input channels
        .a0(active_compute_data[0*D_WIDTH +: D_WIDTH]),
        .a1(active_compute_data[1*D_WIDTH +: D_WIDTH]),
        .a2(active_compute_data[2*D_WIDTH +: D_WIDTH]),
        .a3(active_compute_data[3*D_WIDTH +: D_WIDTH]),
        .a4(active_compute_data[4*D_WIDTH +: D_WIDTH]),
        .a5(active_compute_data[5*D_WIDTH +: D_WIDTH]),
        .a6(active_compute_data[6*D_WIDTH +: D_WIDTH]),
        .a7(active_compute_data[7*D_WIDTH +: D_WIDTH]),
        
        .out_acc(ce_raw_acc)
    );

endmodule